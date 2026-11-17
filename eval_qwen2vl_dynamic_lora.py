import json
import os
import re
import traceback
from collections import defaultdict

import torch
from PIL import Image

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel
from peft.tuners.lora import update_lora_mapping


QWEN_MODEL_ID = os.environ.get("TRAFFIC_QWEN_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
LORA_MODEL_DIR = os.environ.get("TRAFFIC_LORA_DIR", "outputs/qwen2vl_2b_six_task_dynamic_lora")

IMAGE_DIR = os.environ.get("TRAFFIC_IMAGE_DIR", "data/BDD100k/100k-images/test")
LABEL_DIR = os.environ.get("TRAFFIC_LABEL_DIR", "data/BDD100k/100k-n-q-l-six-tasks/test")

MAX_SAMPLES = 100
MAX_NEW_TOKENS = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 768 * 28 * 28

TASK_TO_BLOCK = {
    "Scene Attribute VQA": 0,
    "Object Presence VQA": 1,
    "Lane Presence VQA": 2,
    "Coarse Spatial VQA": 3,
    "Driving Condition VQA": 4,
    "Candidate Box Selection": 5,
}

SUPPORTED_TASK_TYPES = list(TASK_TO_BLOCK.keys())


def load_dataset():
    dataset = []

    if not os.path.exists(LABEL_DIR) or not os.path.exists(IMAGE_DIR):
        print("Image or label directory does not exist.")
        return dataset

    json_files = sorted([f for f in os.listdir(LABEL_DIR) if f.endswith(".json")])

    print(f"Scanning for up to {MAX_SAMPLES} samples...")

    for json_file in json_files:
        if len(dataset) >= MAX_SAMPLES:
            break

        json_path = os.path.join(LABEL_DIR, json_file)

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            img_name = data.get("name", "")
            img_path = os.path.join(IMAGE_DIR, img_name + ".jpg")

            if not os.path.exists(img_path):
                continue

            if "qa_pairs" in data and isinstance(data["qa_pairs"], list):
                qa_pairs = []

                for item in data["qa_pairs"]:
                    if not isinstance(item, dict):
                        continue

                    question = item.get("question", "")
                    label = item.get("label", "")
                    task_type = item.get("task_type", "Unknown")
                    category = item.get("category", "Unknown")

                    if question == "":
                        continue

                    qa_pairs.append({
                        "task_type": task_type,
                        "category": category,
                        "question": question,
                        "label": label
                    })

                if qa_pairs:
                    dataset.append({
                        "name": img_name,
                        "image_path": img_path,
                        "qa_pairs": qa_pairs
                    })

            elif "questions" in data and "labels" in data:
                questions = data["questions"]
                labels = data["labels"]

                if len(questions) != len(labels):
                    continue

                qa_pairs = []
                for q, l in zip(questions, labels):
                    qa_pairs.append({
                        "task_type": "Unknown",
                        "category": "Unknown",
                        "question": q,
                        "label": l
                    })

                dataset.append({
                    "name": img_name,
                    "image_path": img_path,
                    "qa_pairs": qa_pairs
                })

        except Exception as e:
            print(f"[Warning] Could not read {json_file}: {e}")
            continue

    print(f"Loaded {len(dataset)} test samples.\n")
    return dataset


def build_instruction_by_task(task_type: str) -> str:
    if task_type == "Scene Attribute VQA":
        return (
            "You are an expert in autonomous driving scene analysis.\n"
            "Answer the question using only one label from the provided label set.\n"
            "Do not include explanations, prefixes, or extra words."
        )

    elif task_type in [
        "Object Presence VQA",
        "Lane Presence VQA",
        "Driving Condition VQA"
    ]:
        return (
            "You are an expert in autonomous driving scene analysis.\n"
            "Answer the question using only yes or no.\n"
            "Do not include explanations, prefixes, or extra words."
        )

    elif task_type == "Coarse Spatial VQA":
        return (
            "You are an expert in autonomous driving scene analysis.\n"
            "Answer the question using only one of the following labels: "
            "left, center, right, multiple, none.\n"
            "Do not include explanations, prefixes, or extra words."
        )

    elif task_type == "Candidate Box Selection":
        return (
            "You are an expert in autonomous driving scene analysis.\n"
            "Answer using only option letters such as A, B, C, D, or none.\n"
            "If multiple options are correct, separate them with commas, such as A, C.\n"
            "Do not include explanations, prefixes, or extra words."
        )

    else:
        return (
            "You are an expert in autonomous driving scene analysis.\n"
            "Answer the question directly using only the expected label.\n"
            "Do not include explanations, prefixes, or extra words."
        )


def build_single_prompt(question, task_type):
    instruction = build_instruction_by_task(task_type)
    return (
        f"{instruction}\n\n"
        f"Question: {question}\n"
        f"Answer:"
    )


def normalize_text(text):
    if text is None:
        return ""

    text = str(text).strip().lower()
    text = text.rstrip(" .")
    text = re.sub(r"\s+", " ", text)
    return text


def split_multi_label(text):
    text = normalize_text(text)

    if text == "":
        return set()

    parts = [x.strip() for x in text.split(",") if x.strip()]
    return set(parts)


def evaluate_metrics_by_question(all_predictions, all_truths, all_meta):
    if not all_truths or not all_predictions:
        return []

    num_samples = len(all_truths)
    num_questions = min(len(all_truths[0]), len(all_predictions[0]))

    question_metrics = []

    for q_idx in range(num_questions):
        correct = 0
        tp_total = 0
        fp_total = 0
        fn_total = 0

        meta = all_meta[0][q_idx] if all_meta and len(all_meta[0]) > q_idx else {}
        task_type = meta.get("task_type", "Unknown")
        category = meta.get("category", "Unknown")
        question_text = meta.get("question", "")

        valid_count = 0

        for i in range(num_samples):
            if q_idx >= len(all_predictions[i]) or q_idx >= len(all_truths[i]):
                continue

            pred_set = split_multi_label(all_predictions[i][q_idx])
            truth_set = split_multi_label(all_truths[i][q_idx])

            valid_count += 1

            if pred_set == truth_set:
                correct += 1

            tp = len(pred_set.intersection(truth_set))
            fp = len(pred_set - truth_set)
            fn = len(truth_set - pred_set)

            tp_total += tp
            fp_total += fp
            fn_total += fn

        acc = correct / valid_count if valid_count > 0 else 0
        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
        recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        question_metrics.append({
            "Question_Index": q_idx + 1,
            "Task_Type": task_type,
            "Category": category,
            "Question": question_text,
            "Accuracy": round(acc, 4),
            "F1_Score": round(f1, 4)
        })

    return question_metrics


def evaluate_metrics_by_task_type(all_predictions, all_truths, all_meta):
    task_records = defaultdict(list)

    num_samples = len(all_truths)

    for i in range(num_samples):
        num_items = min(len(all_predictions[i]), len(all_truths[i]), len(all_meta[i]))

        for j in range(num_items):
            pred = all_predictions[i][j]
            truth = all_truths[i][j]
            meta = all_meta[i][j]

            task_type = meta.get("task_type", "Unknown")
            task_records[task_type].append((pred, truth))

    task_metrics = {}

    for task_type, items in task_records.items():
        correct = 0
        tp_total = 0
        fp_total = 0
        fn_total = 0

        for pred, truth in items:
            pred_set = split_multi_label(pred)
            truth_set = split_multi_label(truth)

            if pred_set == truth_set:
                correct += 1

            tp = len(pred_set.intersection(truth_set))
            fp = len(pred_set - truth_set)
            fn = len(truth_set - pred_set)

            tp_total += tp
            fp_total += fp
            fn_total += fn

        total = len(items)
        acc = correct / total if total > 0 else 0
        precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
        recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        task_metrics[task_type] = {
            "Count": total,
            "Accuracy": round(acc, 4),
            "F1_Score": round(f1, 4)
        }

    return task_metrics


def evaluate_overall_metrics(all_predictions, all_truths):
    correct = 0
    total = 0
    tp_total = 0
    fp_total = 0
    fn_total = 0

    for sample_preds, sample_truths in zip(all_predictions, all_truths):
        num_items = min(len(sample_preds), len(sample_truths))

        for i in range(num_items):
            pred_set = split_multi_label(sample_preds[i])
            truth_set = split_multi_label(sample_truths[i])

            total += 1

            if pred_set == truth_set:
                correct += 1

            tp = len(pred_set.intersection(truth_set))
            fp = len(pred_set - truth_set)
            fn = len(truth_set - pred_set)

            tp_total += tp
            fp_total += fp
            fn_total += fn

    acc = correct / total if total > 0 else 0
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "Total_QA_Pairs": total,
        "Overall_Accuracy": round(acc, 4),
        "Overall_F1": round(f1, 4)
    }


def load_finetuned_qwen():
    if not os.path.exists(LORA_MODEL_DIR):
        raise FileNotFoundError(f"LoRA directory does not exist: {LORA_MODEL_DIR}")

    print(f"\nLoading processor: {LORA_MODEL_DIR}")
    processor = AutoProcessor.from_pretrained(
        LORA_MODEL_DIR,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        trust_remote_code=True
    )

    print(f"Loading base model: {QWEN_MODEL_ID}")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_ID,
        torch_dtype=torch.float32
    ).to(DEVICE)

    print(f"Loading LoRA adapter: {LORA_MODEL_DIR}")
    model = PeftModel.from_pretrained(
        base_model,
        LORA_MODEL_DIR
    ).to(DEVICE)

    model.eval()

    return model, processor


def call_local_qwen_with_lora(dataset):
    print(f"\nLoading Qwen2-VL with a trained dynamic LoRA adapter...")
    model, processor = load_finetuned_qwen()

    print("Model loaded. Running inference...\n")
    all_predictions = []

    for idx, sample in enumerate(dataset):
        image_path = sample["image_path"]
        qa_pairs = sample["qa_pairs"]

        try:
            pil_image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Could not load image {image_path}: {e}. Using empty answers.")
            all_predictions.append([""] * len(qa_pairs))
            continue

        print(f"\n[{idx + 1}/{len(dataset)}] Processing image: {sample['name']}")
        sample_preds = []

        for q_idx, qa in enumerate(qa_pairs):
            question = qa["question"]
            true_label = qa["label"]
            task_type = qa.get("task_type", "Unknown")
            category = qa.get("category", "Unknown")

            block_id = TASK_TO_BLOCK.get(task_type, 0)


            update_lora_mapping([block_id])

            raw_prompt = build_single_prompt(question, task_type)

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": raw_prompt}
                    ]
                }
            ]

            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = processor(
                text=[text],
                images=[pil_image],
                padding=True,
                return_tensors="pt"
            )

            inputs = inputs.to(DEVICE)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=False
                )

            generated_ids_trimmed = [
                out_ids[len(in_ids):]
                for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]

            generated_text = processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]

            answer = generated_text.strip()

            print(
                f"  -> Q{q_idx + 1:02d} "
                f"[{task_type} | {category}] "
                f"block={block_id} | "
                f"Pred: {answer:<25} | label: {true_label}"
            )

            sample_preds.append(answer)

        all_predictions.append(sample_preds)

    del model
    del processor
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    return all_predictions


def main():
    dataset = load_dataset()
    if not dataset:
        print("No samples found for evaluation.")
        return

    all_truths = []
    all_meta = []

    for sample in dataset:
        sample_truths = []
        sample_meta = []

        for qa in sample["qa_pairs"]:
            sample_truths.append(qa["label"])
            sample_meta.append({
                "task_type": qa.get("task_type", "Unknown"),
                "category": qa.get("category", "Unknown"),
                "question": qa.get("question", "")
            })

        all_truths.append(sample_truths)
        all_meta.append(sample_meta)

    results_summary = {}

    model_callers = {
        "Qwen2-VL-2B-Instruct+DynamicLoRA": call_local_qwen_with_lora,
    }

    for model_name, caller_func in model_callers.items():
        try:
            all_predictions = caller_func(dataset)

            if not all_predictions:
                continue

            overall_metrics = evaluate_overall_metrics(all_predictions, all_truths)
            per_task_metrics = evaluate_metrics_by_task_type(all_predictions, all_truths, all_meta)
            per_question_metrics = evaluate_metrics_by_question(all_predictions, all_truths, all_meta)

            results_summary[model_name] = {
                "Overall": overall_metrics,
                "Per_Task_Type_Metrics": per_task_metrics,
                "Per_Question_Metrics": per_question_metrics
            }

            print(f"\n[{model_name}] Evaluation complete.")
            print(json.dumps(results_summary[model_name], indent=4, ensure_ascii=False))

        except Exception:
            print(f"\n[{model_name}] Evaluation failed:")
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("Evaluation summary")
    print("=" * 80)
    print(json.dumps(results_summary, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
