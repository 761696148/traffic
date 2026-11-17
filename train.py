import os
import json
import random
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any

import torch
from PIL import Image
from tqdm import tqdm

from transformers import AutoModelForCausalLM, AutoProcessor
from peft import get_peft_model, LoraConfig
from peft.tuners.lora import update_lora_mapping


PHI35_MODEL_ID = os.environ.get("TRAFFIC_PHI_MODEL", "microsoft/Phi-3.5-vision-instruct")

IMAGE_DIR = os.environ.get("TRAFFIC_IMAGE_DIR", "data/BDD100k/100k-images/test")
LABEL_DIR = os.environ.get("TRAFFIC_LABEL_DIR", "data/BDD100k/100k-n-q-l-six-tasks/test")

SAVE_DIR = os.environ.get("TRAFFIC_SAVE_DIR", "outputs/phi35_six_task_dynamic_lora")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_SAMPLES = 1000
NUM_EPOCHS = 1
LEARNING_RATE = 2e-4
MAX_NEW_TOKENS = 32


NUM_BLOCK = 6
NUM_RANK_PER_BLOCK = 4
TOTAL_LORA_RANK = NUM_BLOCK * NUM_RANK_PER_BLOCK


TASK_TO_BLOCK = {
    "Scene Attribute VQA": 0,
    "Object Presence VQA": 1,
    "Lane Presence VQA": 2,
    "Coarse Spatial VQA": 3,
    "Driving Condition VQA": 4,
    "Candidate Box Selection": 5,
}

SUPPORTED_TASK_TYPES = list(TASK_TO_BLOCK.keys())


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


def build_phi_prompt_without_answer(question: str, task_type: str) -> str:


    instruction = build_instruction_by_task(task_type)

    prompt = (
        f"<|user|>\n"
        f"<|image_1|>\n"
        f"{instruction}\n\n"
        f"Question: {question}\n"
        f"Answer:"
        f"<|end|>\n"
        f"<|assistant|>\n"
    )

    return prompt


def build_phi_prompt_with_answer(question: str, answer: str, task_type: str) -> str:


    prompt_without_answer = build_phi_prompt_without_answer(question, task_type)

    full_prompt = (
        f"{prompt_without_answer}"
        f"{answer}"
        f"<|end|>\n"
    )

    return full_prompt


def load_six_task_dataset(
    image_dir: str,
    label_dir: str,
    max_samples: int = None
) -> List[Dict[str, Any]]:


    samples = []

    json_files = sorted([
        f for f in os.listdir(label_dir)
        if f.endswith(".json")
    ])

    image_count = 0

    for json_file in json_files:
        if max_samples is not None and image_count >= max_samples:
            break

        json_path = os.path.join(label_dir, json_file)

        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"[Warning] Could not read: {json_path}, {e}")
            continue

        image_name = data.get("name", Path(json_file).stem)
        image_path = os.path.join(image_dir, image_name + ".jpg")

        if not os.path.exists(image_path):
            continue

        qa_pairs = data.get("qa_pairs", [])
        if not qa_pairs:
            continue

        image_count += 1

        for qa in qa_pairs:
            task_type = qa.get("task_type", "Unknown")
            category = qa.get("category", "Unknown")
            question = qa.get("question", "")
            answer = qa.get("label", "")

            if task_type not in TASK_TO_BLOCK:
                continue

            if question == "" or answer == "":
                continue

            block_id = TASK_TO_BLOCK[task_type]

            samples.append({
                "image_path": image_path,
                "image_name": image_name,
                "task_type": task_type,
                "category": category,
                "question": question,
                "answer": str(answer),
                "block_id": block_id
            })

    print(f"Images loaded: {image_count}")
    print(f"Question-answer pairs loaded: {len(samples)}")

    return samples


def build_training_inputs(
    processor,
    sample: Dict[str, Any],
    device: str = DEVICE
):


    image = Image.open(sample["image_path"]).convert("RGB")

    question = sample["question"]
    answer = sample["answer"]
    task_type = sample["task_type"]

    prompt_without_answer = build_phi_prompt_without_answer(
        question=question,
        task_type=task_type
    )

    full_prompt = build_phi_prompt_with_answer(
        question=question,
        answer=answer,
        task_type=task_type
    )

    inputs = processor(
        full_prompt,
        [image],
        return_tensors="pt"
    )

    prompt_inputs = processor(
        prompt_without_answer,
        [image],
        return_tensors="pt"
    )

    inputs = inputs.to(device)
    prompt_inputs = prompt_inputs.to(device)

    labels = inputs["input_ids"].clone()
    prompt_len = prompt_inputs["input_ids"].shape[1]
    labels[:, :prompt_len] = -100

    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is not None:
        labels[labels == pad_token_id] = -100

    inputs["labels"] = labels


    # if device == "cuda":
    #     for k, v in inputs.items():
    #         if hasattr(v, "dtype") and v.dtype == torch.float32:
    #             inputs[k] = v.to(torch.float16)

    return inputs


def print_trainable_parameters(model):
    trainable = 0
    total = 0

    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad:
            trainable += param.numel()

    print(
        f"Trainable params: {trainable} | "
        f"Total params: {total} | "
        f"Trainable ratio: {100 * trainable / total:.4f}%"
    )


def load_phi35_with_dynamic_lora():
    print(f"Loading processor: {PHI35_MODEL_ID}")
    processor = AutoProcessor.from_pretrained(
        PHI35_MODEL_ID,
        trust_remote_code=True
    )

    print(f"Loading Phi-3.5-Vision: {PHI35_MODEL_ID}")
    base_model = AutoModelForCausalLM.from_pretrained(
        PHI35_MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        _attn_implementation="eager"
    ).to(DEVICE)

    base_model.train()

    target_modules = [
        "qkv_proj",
        "o_proj",
        "gate_up_proj",
        "down_proj",
    ]

    lora_config = LoraConfig(
        r=TOTAL_LORA_RANK,
        lora_alpha=TOTAL_LORA_RANK,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        fan_in_fan_out=False,
        num_rank_per_block=NUM_RANK_PER_BLOCK,
    )

    print("Adding dynamic LoRA adapters...")
    model = get_peft_model(base_model, lora_config)

    model.to(DEVICE)
    model.train()

    print_trainable_parameters(model)

    return model, processor


def train():
    os.makedirs(SAVE_DIR, exist_ok=True)

    samples = load_six_task_dataset(
        image_dir=IMAGE_DIR,
        label_dir=LABEL_DIR,
        max_samples=MAX_SAMPLES
    )

    if not samples:
        print("No training samples found.")
        return

    random.shuffle(samples)

    model, processor = load_phi35_with_dynamic_lora()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE
    )

    global_step = 0

    for epoch in range(NUM_EPOCHS):
        print(f"\n========== Epoch {epoch + 1}/{NUM_EPOCHS} ==========")

        random.shuffle(samples)

        pbar = tqdm(samples)

        for sample in pbar:
            task_type = sample["task_type"]
            block_id = sample["block_id"]


            update_lora_mapping([block_id])

            try:
                inputs = build_training_inputs(
                    processor=processor,
                    sample=sample,
                    device=DEVICE
                )

                outputs = model(**inputs)
                loss = outputs.loss

                loss.backward()

                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    max_norm=1.0
                )

                optimizer.step()
                optimizer.zero_grad()

                global_step += 1

                pbar.set_description(
                    f"loss={loss.item():.4f} | "
                    f"task={task_type} | "
                    f"block={block_id}"
                )

                if global_step % 50 == 0:
                    print(
                        f"\nStep {global_step} | "
                        f"Loss: {loss.item():.4f} | "
                        f"Task: {task_type} | "
                        f"Block: {block_id} | "
                        f"Q: {sample['question']} | "
                        f"A: {sample['answer']}"
                    )

            except RuntimeError as e:
                print(f"\n[RuntimeError] step={global_step}, task={task_type}, block={block_id}")
                print(e)

                optimizer.zero_grad()

                if DEVICE == "cuda":
                    torch.cuda.empty_cache()

                continue

            except Exception as e:
                print(f"\n[Error] step={global_step}, task={task_type}, block={block_id}")
                print(e)

                optimizer.zero_grad()
                continue

    print(f"\nTraining complete. Saving LoRA to: {SAVE_DIR}")
    model.save_pretrained(SAVE_DIR)
    processor.save_pretrained(SAVE_DIR)

    print("Saved.")


if __name__ == "__main__":
    train()
