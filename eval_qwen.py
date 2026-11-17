import json
import os
import re
import traceback
import time
from collections import defaultdict
import random

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel
from peft.tuners.lora import update_lora_mapping


QWEN_MODEL_ID = os.environ.get("TRAFFIC_QWEN_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
LORA_MODEL_DIR = os.environ.get("TRAFFIC_LORA_DIR", "outputs/qwen2vl_2b_causaldrive_dynamic_lora")


CAUSALSCENE_BASE_DIR = os.environ.get("TRAFFIC_CAUSALSCENE_ROOT", "data/CAUSALSCENE")

USE_LORA = os.environ.get("TRAFFIC_USE_LORA", "0") == "1"

MAX_SAMPLES_PER_TASK = 50
MAX_NEW_TOKENS = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 768 * 28 * 28

TASK_TO_BLOCK = {"SEP": 0, "DHO": 1, "RCE": 2, "ADR": 3, "MTA": 4}


def clean_string(s):
    s = str(s).strip()
    return re.sub(r'[\*\n\"\'\[\]]', '', s).lower()


def parse_llm_mcq_answer(prediction, correct_label, question_text):

    clean_pred = clean_string(prediction)
    clean_label = clean_string(correct_label)

    if clean_pred == clean_label:
        return True

    if clean_label in clean_pred:
        pattern = re.compile(r"\[(.*?)\]")
        matches = pattern.findall(question_text)
        if matches:
            options_str = matches[-1]
            options = [clean_string(opt) for opt in options_str.split("', '")]
            other_options_in_pred = [opt for opt in options if opt != clean_label and opt in clean_pred]
            if not other_options_in_pred:
                return True
    return False


def scan_and_load_tasks():
    tasks_paths = {
        "SEP": os.path.join(CAUSALSCENE_BASE_DIR, "SEP"),
        "RCE": os.path.join(CAUSALSCENE_BASE_DIR, "RCE"),
        "ADR": os.path.join(CAUSALSCENE_BASE_DIR, "ADR"),
        "DHO_Base": os.path.join(CAUSALSCENE_BASE_DIR, "DHO", "bdd"),
        "DHO_Corner": os.path.join(CAUSALSCENE_BASE_DIR, "DHO", "Corner_Cases"),
        "MTA": os.path.join(CAUSALSCENE_BASE_DIR, "MTA")
    }

    dataset_dict = defaultdict(list)

    for task_key, folder_path in tasks_paths.items():
        img_dir = os.path.join(folder_path, "images")
        lbl_dir = os.path.join(folder_path, "label")

        if not os.path.exists(lbl_dir):
            print(f"[Warning] Skipping missing directory: {lbl_dir}")
            continue

        json_files = sorted([f for f in os.listdir(lbl_dir) if f.endswith(".json")])
        random.seed(42)
        random.shuffle(json_files)

        count = 0
        for json_file in json_files:
            if count >= MAX_SAMPLES_PER_TASK: break

            with open(os.path.join(lbl_dir, json_file), "r", encoding="utf-8") as f:
                data = json.load(f)

            img_name = data.get("name", json_file.replace(".json", ""))
            img_path = os.path.join(img_dir, f"{img_name}.jpg")

            if not os.path.exists(img_path): continue

            qa_pairs = data.get("qa_pairs", [])
            if not qa_pairs: continue
            core_task_type = qa_pairs[0].get("task_type", "SEP")

            dataset_dict[task_key].append({
                "image_path": img_path,
                "image_name": img_name,
                "qa_pairs": qa_pairs,
                "task_key": task_key,
                "block_id": TASK_TO_BLOCK.get(core_task_type, 0)
            })
            count += 1

        print(f"[{task_key}] Loaded {len(dataset_dict[task_key])} samples.")

    return dataset_dict


def build_single_prompt(question):
    return (
        "You are an expert autonomous driving agent.\n"
        "Please select the most appropriate exact phrase from the provided options to answer the question.\n"
        "Output ONLY the exact phrase from the options. Do not include extra words.\n\n"
        f"Question: {question}\nAnswer:"
    )


def run_inference_and_evaluate(dataset_dict, use_lora):
    print(f"\nLoading processor and model in bfloat16...")
    processor = AutoProcessor.from_pretrained(
        LORA_MODEL_DIR if use_lora and os.path.exists(LORA_MODEL_DIR) else QWEN_MODEL_ID,
        min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS, trust_remote_code=True
    )


    model = Qwen2VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_ID, torch_dtype=torch.bfloat16
    ).to(DEVICE)

    if use_lora:
        print(f"Loading the task-routed MoE-LoRA adapter...")
        model = PeftModel.from_pretrained(model, LORA_MODEL_DIR).to(DEVICE)
    model.eval()

    task_scores = {k: {"score": 0.0, "total": 0} for k in dataset_dict.keys()}

    print("\nRunning inference and scoring by task...")

    for task_key, samples in dataset_dict.items():
        print(f"\n" + "=" * 50)
        print(f" Evaluating task: {task_key}")
        print("=" * 50)

        for idx, sample in enumerate(samples):
            image_path = sample["image_path"]
            qa_pairs = sample["qa_pairs"]
            block_id = sample["block_id"]

            if use_lora: update_lora_mapping([block_id])

            try:
                pil_image = Image.open(image_path).convert("RGB")
            except Exception:
                continue

            print(f"\n📸 [Image {idx + 1}/{len(samples)}] {sample['image_name']}.jpg")
            img_results = {}

            for qa in qa_pairs:
                question = qa["question"]
                true_label = qa.get("label", "")
                category = qa.get("category", "unknown")

                raw_prompt = build_single_prompt(question)
                messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": raw_prompt}]}]
                text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                inputs = processor(text=[text], images=[pil_image], padding=True, return_tensors="pt").to(DEVICE)


                t0 = time.time()


                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        top_k=None
                    )

                cost_time = time.time() - t0

                generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in
                                         zip(inputs["input_ids"], generated_ids)]
                answer = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True,
                                                clean_up_tokenization_spaces=False)[0].strip()

                is_correct = parse_llm_mcq_answer(answer, true_label, question)
                img_results[category] = is_correct


                mark = "✅" if is_correct else "❌"
                print(f"  [{category}] ({cost_time:.2f}s) {mark}")
                print(f"    - Pred : {answer}")
                print(f"    - Label: {true_label}")


            task_scores[task_key]["total"] += 1

            if task_key == "SEP":
                if len(img_results) > 0: task_scores["SEP"]["score"] += sum(img_results.values()) / len(img_results)
            elif task_key in ["DHO_Base", "DHO_Corner"]:
                if len(img_results) > 0 and all(img_results.values()): task_scores[task_key]["score"] += 1.0
            elif task_key == "RCE":
                row_corr = img_results.get("right_of_way", False)
                sig_corr = img_results.get("signal_constraint", False)
                task_scores["RCE"]["score"] += (0.7 * float(row_corr)) + (0.3 * float(sig_corr))
            elif task_key == "ADR":
                if len(img_results) >= 3 and all(img_results.values()): task_scores["ADR"]["score"] += 1.0
            elif task_key == "MTA":
                if len(img_results) >= 4 and all(img_results.values()): task_scores["MTA"]["score"] += 1.0

    return task_scores


def print_report(task_scores):
    report = "\n" + "=" * 70 + "\n"
    report += "🚗 CausalScene 6-Branch Strict Evaluation Report\n"
    report += "=" * 70 + "\n\n"

    def get_acc(task_key):
        total = task_scores.get(task_key, {}).get("total", 0)
        if total == 0: return 0.0
        return task_scores[task_key]["score"] / total

    report += "[1] SEP Score (Independent Average Accuracy)\n"
    report += f"    - Static Perception:               {get_acc('SEP'):.2%} (N={task_scores.get('SEP', {}).get('total', 0)})\n\n"

    report += "[2] RCE Score (Weighted Accuracy: 70% RoW, 30% Signal)\n"
    report += f"    - Regulatory Constraints:          {get_acc('RCE'):.2%} (N={task_scores.get('RCE', {}).get('total', 0)})\n\n"

    report += "[3] DHO Scores (Strict Joint Accuracy - Zero Misses Allowed)\n"
    report += f"    - Base Hazards (BDD100k):          {get_acc('DHO_Base'):.2%} (N={task_scores.get('DHO_Base', {}).get('total', 0)})\n"
    report += f"    - Corner Case Hazards (CODA):      {get_acc('DHO_Corner'):.2%} (N={task_scores.get('DHO_Corner', {}).get('total', 0)})\n\n"

    report += "[4] ADR Score (Strict Synergy Score)\n"
    report += f"    - Action + Focus + Rationale:      {get_acc('ADR'):.2%} (N={task_scores.get('ADR', {}).get('total', 0)})\n\n"

    report += "[5] MTA Score (V2X Broadcast Synergy Score)\n"
    report += f"    - Macroscopic Assessment + V2X:    {get_acc('MTA'):.2%} (N={task_scores.get('MTA', {}).get('total', 0)})\n\n"

    report += "-" * 70 + "\n"

    accs = [get_acc(k) for k in task_scores.keys() if task_scores[k]["total"] > 0]
    overall = sum(accs) / len(accs) if accs else 0.0

    report += f"[🏆] Overall CausalScene Score (Macro Avg): {overall * 100:.2f} / 100\n"
    report += "=" * 70 + "\n"

    print(report)


if __name__ == "__main__":
    dataset_dict = scan_and_load_tasks()
    if not dataset_dict:
        print("No test samples found.")
        exit()

    try:
        final_scores = run_inference_and_evaluate(dataset_dict, USE_LORA)
        print_report(final_scores)
    except Exception:
        traceback.print_exc()