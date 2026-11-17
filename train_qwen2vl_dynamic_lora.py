import os
import json
import random
from pathlib import Path
from typing import List, Dict, Any

import torch
from PIL import Image
from tqdm import tqdm

from transformers import AutoProcessor, Qwen2VLForConditionalGeneration, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training
from peft.tuners.lora import update_lora_mapping


QWEN_MODEL_ID = os.environ.get("TRAFFIC_QWEN_MODEL", "Qwen/Qwen2-VL-2B-Instruct")


IMAGE_DIR = os.environ.get("TRAFFIC_IMAGE_DIR", "data/CAUSALSCENE")
LABEL_DIR = os.environ.get("TRAFFIC_LABEL_DIR", "data/CAUSALSCENE")
SAVE_DIR = os.environ.get("TRAFFIC_SAVE_DIR", "outputs/qwen2vl_2b_causaldrive_dynamic_lora")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MAX_SAMPLES = 5000
NUM_EPOCHS = 2
LEARNING_RATE = 2e-4
GRADIENT_ACCUMULATION_STEPS = 4


MIN_PIXELS = 256 * 28 * 28
MAX_PIXELS = 768 * 28 * 28


NUM_BLOCK = 5
NUM_RANK_PER_BLOCK = 16
TOTAL_LORA_RANK = NUM_BLOCK * NUM_RANK_PER_BLOCK

TASK_TO_BLOCK = {
    "SEP": 0,  # Static Environment Perception
    "DHO": 1,  # Dynamic Hazard Observation
    "RCE": 2,  # Regulatory Constraint Evaluation
    "ADR": 3, # Action & Decision Rationale
    "MTA": 4
}


def build_instruction() -> str:


    return (
        "You are an expert autonomous driving cognitive coordinator.\n"
        "Please select the most appropriate exact phrase from the provided options to answer the question.\n"
        "Output ONLY the exact phrase from the options. Do not include any prefix, explanation, or punctuation outside the phrase."
    )


def build_qwen_prompt(question: str, answer: str, processor, include_answer: bool = True) -> str:
    user_text = f"{build_instruction()}\n\nQuestion: {question}\nAnswer:"

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": user_text},
            ],
        }
    ]

    if include_answer:
        messages.append({
            "role": "assistant",
            "content": [{"type": "text", "text": str(answer)}]
        })
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    else:
        return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def load_causaldrive_dataset(image_dir: str, label_dir: str, max_samples: int = None) -> List[Dict[str, Any]]:
    samples = []
    if not os.path.exists(image_dir) or not os.path.exists(label_dir):
        raise FileNotFoundError("Image or label directory does not exist.")

    json_files = sorted([f for f in os.listdir(label_dir) if f.endswith(".json")])
    image_count = 0

    print("Loading CausalDrive question-answer pairs...")
    for json_file in json_files:
        if max_samples is not None and image_count >= max_samples:
            break

        json_path = os.path.join(label_dir, json_file)
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            continue

        image_name = data.get("name", Path(json_file).stem)
        image_path = os.path.join(image_dir, image_name + ".jpg")
        if not os.path.exists(image_path):
            continue

        qa_pairs = data.get("qa_pairs", [])
        has_valid_qa = False

        for qa in qa_pairs:
            task_type = qa.get("task_type")
            if task_type not in TASK_TO_BLOCK:
                continue

            block_id = qa.get("block_id", TASK_TO_BLOCK[task_type])
            samples.append({
                "image_path": image_path,
                "task_type": task_type,
                "category": qa.get("category"),
                "question": qa.get("question", ""),
                "answer": str(qa.get("label", "")),
                "block_id": block_id,
            })
            has_valid_qa = True

        if has_valid_qa:
            image_count += 1

    print(f"Loaded {image_count} images and {len(samples)} training examples.")
    return samples


def build_training_inputs(processor, sample: Dict[str, Any], device: str):
    image = Image.open(sample["image_path"]).convert("RGB")

    prompt_no_ans = build_qwen_prompt(sample["question"], "", processor, include_answer=False)
    prompt_with_ans = build_qwen_prompt(sample["question"], sample["answer"], processor, include_answer=True)

    inputs = processor(
        text=[prompt_with_ans],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(device)

    prompt_inputs = processor(
        text=[prompt_no_ans],
        images=[image],
        padding=True,
        return_tensors="pt",
    ).to(device)


    labels = inputs["input_ids"].clone()
    prompt_len = prompt_inputs["input_ids"].shape[1]
    labels[:, :prompt_len] = -100

    if processor.tokenizer.pad_token_id is not None:
        labels[labels == processor.tokenizer.pad_token_id] = -100

    inputs["labels"] = labels
    return inputs


def print_gpu_memory():
    if DEVICE == "cuda":
        allocated = torch.cuda.memory_allocated() / 1024 ** 3
        reserved = torch.cuda.memory_reserved() / 1024 ** 3
        print(f"    [GPU Mem] Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")


def load_qwen2vl_moe_lora():
    print(f"Loading processor...")
    processor = AutoProcessor.from_pretrained(
        QWEN_MODEL_ID,
        min_pixels=MIN_PIXELS,
        max_pixels=MAX_PIXELS,
        trust_remote_code=True,
    )

    print(f"Loading Qwen2-VL-2B in bfloat16...")
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )


    base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    base_model.enable_input_require_grads()

    base_model.config.use_cache = False
    base_model = prepare_model_for_kbit_training(base_model)

    if hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})


    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

    lora_config = LoraConfig(
        r=TOTAL_LORA_RANK,
        lora_alpha=TOTAL_LORA_RANK * 2,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        num_rank_per_block=NUM_RANK_PER_BLOCK,
    )

    print("Adding task-routed MoE-LoRA adapters...")
    model = get_peft_model(base_model, lora_config)
    model.train()

    trainable, total = 0, 0
    for _, param in model.named_parameters():
        total += param.numel()
        if param.requires_grad: trainable += param.numel()

    print(f"MoE-LoRA parameters: Trainable={trainable} ({100 * trainable / total:.4f}%) | Total={total}")
    return model, processor


def train():
    os.makedirs(SAVE_DIR, exist_ok=True)
    samples = load_causaldrive_dataset(IMAGE_DIR, LABEL_DIR, MAX_SAMPLES)
    if not samples: return

    model, processor = load_qwen2vl_moe_lora()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )

    global_step = 0
    optimizer.zero_grad()

    for epoch in range(NUM_EPOCHS):
        print(f"\n" + "=" * 50)
        print(f" 🚀 Epoch {epoch + 1} / {NUM_EPOCHS}")
        print(f"==================================================")


        random.shuffle(samples)

        pbar = tqdm(samples, desc=f"Epoch {epoch + 1}")
        accumulated_loss = 0.0

        for idx, sample in enumerate(pbar):
            task_type = sample["task_type"]
            block_id = sample["block_id"]


            update_lora_mapping([block_id])

            try:
                inputs = build_training_inputs(processor, sample, DEVICE)
                outputs = model(**inputs)


                loss = outputs.loss / GRADIENT_ACCUMULATION_STEPS
                loss.backward()
                accumulated_loss += loss.item()

                if (idx + 1) % GRADIENT_ACCUMULATION_STEPS == 0:

                    torch.nn.utils.clip_grad_norm_(
                        filter(lambda p: p.requires_grad, model.parameters()),
                        max_norm=1.0
                    )
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    pbar.set_postfix({"loss": f"{accumulated_loss:.4f}", "task": task_type})
                    accumulated_loss = 0.0

                    if global_step % 100 == 0:
                        print(
                            f"\n[Step {global_step}] Task: {task_type} (Block {block_id}) | Q: {sample['question'][:60]}... | A: {sample['answer']}")
                        print_gpu_memory()

            except torch.cuda.OutOfMemoryError:
                print(f"\n[OOM] Skipping image after CUDA ran out of memory: {sample['image_path']}")
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                continue
            except Exception as e:
                print(f"\n[Error] Step {global_step}, Task: {task_type} -> {e}")
                optimizer.zero_grad(set_to_none=True)
                continue


        ckpt_dir = os.path.join(SAVE_DIR, f"checkpoint_epoch_{epoch + 1}")
        model.save_pretrained(ckpt_dir)
        processor.save_pretrained(ckpt_dir)

    print(f"\nTraining complete. Adapter saved to: {SAVE_DIR}")
    model.save_pretrained(SAVE_DIR)
    processor.save_pretrained(SAVE_DIR)


if __name__ == "__main__":
    train()