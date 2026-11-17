import json
import os
import re
import traceback
from collections import defaultdict

import torch
from PIL import Image


PHI35_MODEL_ID = os.environ.get("TRAFFIC_PHI_MODEL", "microsoft/Phi-3.5-vision-instruct")


IMAGE_DIR = os.environ.get("TRAFFIC_IMAGE_DIR", "data/BDD100k/100k-images/test")
LABEL_DIR = os.environ.get("TRAFFIC_LABEL_DIR", "data/BDD100k/100k-n-q-l-six-tasks/test")

MAX_SAMPLES = 100
MAX_NEW_TOKENS = 32
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

SUPPORTED_TASK_TYPES = [
    "Scene Attribute VQA",
    "Object Presence VQA",
    "Lane Presence VQA",
    "Coarse Spatial VQA",
    "Driving Condition VQA",
    "Candidate Box Selection"
]


def load_dataset():


    dataset = []

    if not os.path.exists(LABEL_DIR):
        print(f"Label directory does not exist: {LABEL_DIR}")
        return dataset

    if not os.path.exists(IMAGE_DIR):
        print(f"Image directory does not exist: {IMAGE_DIR}")
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


def build_single_prompt(question, task_type=None):


    if task_type == "Scene Attribute VQA":
        instruction = (
            "You are an expert in autonomous driving scene analysis.\n"
            "Answer the question using only one label from the provided label set.\n"
            "Do not include explanations, prefixes, or extra words.\n"
        )

    elif task_type in [
        "Object Presence VQA",
        "Lane Presence VQA",
        "Driving Condition VQA"
    ]:
        instruction = (
            "You are an expert in autonomous driving scene analysis.\n"
            "Answer the question using only yes or no.\n"
            "Do not include explanations, prefixes, or extra words.\n"
        )

    elif task_type == "Coarse Spatial VQA":
        instruction = (
            "You are an expert in autonomous driving scene analysis.\n"
            "Answer the question using only one of the following labels: "
            "left, center, right, multiple, none.\n"
            "Do not include explanations, prefixes, or extra words.\n"
        )

    elif task_type == "Candidate Box Selection":
        instruction = (
            "You are an expert in autonomous driving scene analysis.\n"
            "Answer using only option letters such as A, B, C, D, or none.\n"
            "If multiple options are correct, separate them with commas, such as A, C.\n"
            "Do not include explanations, prefixes, or extra words.\n"
        )

    else:
        instruction = (
            "You are an expert in autonomous driving scene analysis.\n"
            "Answer the question directly using only the expected labels.\n"
            "If multiple labels apply, output them as a comma-separated list.\n"
            "Do not include explanations, prefixes, or extra words.\n"
        )

    prompt = (
        f"{instruction}\n"
        f"Question: {question}\n"
        f"Answer:"
    )

    return prompt


def normalize_text(text):


    if text is None:
        return ""

    text = str(text).strip().lower()


    text = re.sub(r"^answer\s*:\s*", "", text)
    text = re.sub(r"^the answer is\s*", "", text)


    text = text.rstrip(" .")


    text = re.sub(r"\s+", " ", text)

    return text


def split_multi_label(text):


    text = normalize_text(text)

    if text == "":
        return set()

    parts = [x.strip() for x in text.split(",") if x.strip()]
    return set(parts)


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
    precision = tp_total / (tp_total + fp_total) if tp_total + fp_total > 0 else 0
    recall = tp_total / (tp_total + fn_total) if tp_total + fn_total > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

    return {
        "Total_QA_Pairs": total,
        "Overall_Accuracy": round(acc, 4),
        "Overall_F1": round(f1, 4)
    }


def evaluate_metrics_by_task_type(all_predictions, all_truths, all_meta):


    task_records = defaultdict(list)

    num_samples = len(all_truths)

    for i in range(num_samples):
        num_items = min(
            len(all_predictions[i]),
            len(all_truths[i]),
            len(all_meta[i])
        )

        for j in range(num_items):
            pred = all_predictions[i][j]
            truth = all_truths[i][j]
            task_type = all_meta[i][j].get("task_type", "Unknown")

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
        precision = tp_total / (tp_total + fp_total) if tp_total + fp_total > 0 else 0
        recall = tp_total / (tp_total + fn_total) if tp_total + fn_total > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        task_metrics[task_type] = {
            "Count": total,
            "Accuracy": round(acc, 4),
            "F1_Score": round(f1, 4)
        }

    return task_metrics


def evaluate_metrics_by_question(all_predictions, all_truths, all_meta):


    if not all_truths or not all_predictions:
        return []

    num_samples = len(all_truths)
    num_questions = min(len(all_truths[0]), len(all_predictions[0]))

    question_metrics = []

    for q_idx in range(num_questions):
        correct = 0
        valid_count = 0

        tp_total = 0
        fp_total = 0
        fn_total = 0

        meta = all_meta[0][q_idx] if all_meta and len(all_meta[0]) > q_idx else {}
        task_type = meta.get("task_type", "Unknown")
        category = meta.get("category", "Unknown")
        question_text = meta.get("question", "")

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
        precision = tp_total / (tp_total + fp_total) if tp_total + fp_total > 0 else 0
        recall = tp_total / (tp_total + fn_total) if tp_total + fn_total > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        question_metrics.append({
            "Question_Index": q_idx + 1,
            "Task_Type": task_type,
            "Category": category,
            "Question": question_text,
            "Accuracy": round(acc, 4),
            "F1_Score": round(f1, 4)
        })

    return question_metrics


def call_local_phi35_vision(dataset):


    if DEVICE != "cuda":
        print("[Warning] CUDA is unavailable. Phi-3.5-vision-instruct inference typically uses a GPU.")

    print(f"\nLoading Phi-3.5-vision-instruct ({PHI35_MODEL_ID})...")

    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(
        PHI35_MODEL_ID,
        trust_remote_code=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        PHI35_MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16 if DEVICE == "cuda" else torch.float32,
        _attn_implementation="eager"
    ).to(DEVICE).eval()

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

            raw_prompt = build_single_prompt(
                question=question,
                task_type=task_type
            )


            prompt = (
                f"<|user|>\n"
                f"<|image_1|>\n"
                f"{raw_prompt}"
                f"<|end|>\n"
                f"<|assistant|>\n"
            )

            try:
                inputs = processor(
                    prompt,
                    [pil_image],
                    return_tensors="pt"
                )

                inputs = inputs.to(DEVICE)


                if DEVICE == "cuda":
                    for k, v in inputs.items():
                        if hasattr(v, "dtype") and v.dtype == torch.float32:
                            inputs[k] = v.to(torch.float16)

                with torch.no_grad():
                    generated_ids = model.generate(
                        **inputs,
                        max_new_tokens=MAX_NEW_TOKENS,
                        do_sample=False,
                        eos_token_id=processor.tokenizer.eos_token_id
                    )

                input_len = inputs["input_ids"].shape[1]
                generated_ids_trimmed = generated_ids[0][input_len:]

                answer = processor.tokenizer.decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True
                ).strip()

            except Exception as e:
                print(f"  -> Q{q_idx + 1:02d} Inference failed: {e}")
                answer = ""

            print(
                f"  -> Q{q_idx + 1:02d} "
                f"[{task_type} | {category}] "
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
        "Phi-3.5-vision-instruct": call_local_phi35_vision
    }

    for model_name, caller_func in model_callers.items():
        try:
            all_predictions = caller_func(dataset)

            if not all_predictions:
                continue

            overall_metrics = evaluate_overall_metrics(
                all_predictions,
                all_truths
            )

            per_task_metrics = evaluate_metrics_by_task_type(
                all_predictions,
                all_truths,
                all_meta
            )

            per_question_metrics = evaluate_metrics_by_question(
                all_predictions,
                all_truths,
                all_meta
            )

            results_summary[model_name] = {
                "Overall": overall_metrics,
                "Per_Task_Type_Metrics": per_task_metrics,
                "Per_Question_Metrics": per_question_metrics
            }

            print(f"\n[{model_name}] Evaluation complete.")
            print(json.dumps(
                results_summary[model_name],
                indent=4,
                ensure_ascii=False
            ))

        except Exception:
            print(f"\n[{model_name}] Evaluation failed:")
            traceback.print_exc()

    print("\n" + "=" * 80)
    print("Evaluation summary")
    print("=" * 80)

    print(json.dumps(
        results_summary,
        indent=4,
        ensure_ascii=False
    ))


if __name__ == "__main__":
    main()
