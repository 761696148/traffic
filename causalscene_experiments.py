"""Reproducible CausalScene comparative experiments.

The script implements four adapter comparison settings:
1. Zero-shot Qwen2-VL-2B
2. Shared LoRA, rank 16
3. Shared LoRA, rank 80 (parameter-budget control)
4. Five-block task-routed MoE-LoRA, rank 16 per block

No metric is fabricated. ``compare --reserve-only`` writes an empty result
template, while missing adapter checkpoints are recorded as
``pending_checkpoint`` with blank metric cells.

The ``compare-models`` subcommand adds cross-model zero-shot baselines through
three inference backends (standard Transformers chat templates, InternVL and
MiniCPM-V).  Its default registry covers Qwen2-VL, Qwen2.5-VL, InternVL2.5,
MiniCPM-V 2.6 and LLaVA-1.5, and can be replaced with a JSON registry.
"""

from __future__ import annotations

import argparse
import ast
import concurrent.futures
import csv
import gc
import hashlib
import json
import math
import os
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator, Sequence


DEFAULT_DATASET_ROOT = Path(os.environ.get("TRAFFIC_CAUSALSCENE_ROOT", "data/CAUSALSCENE"))
DEFAULT_BASE_MODEL = os.environ.get("TRAFFIC_QWEN_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
DEFAULT_OUTPUT_ROOT = Path(os.environ.get("TRAFFIC_RESULTS_DIR", "outputs/comparison_results"))

BRANCHES = {
    "SEP": ("SEP",),
    "DHO_Base": ("DHO", "bdd"),
    "DHO_Corner": ("DHO", "Corner_Cases"),
    "RCE": ("RCE",),
    "ADR": ("ADR",),
    "MTA": ("MTA",),
}
TASK_TO_BLOCK = {"SEP": 0, "DHO": 1, "RCE": 2, "ADR": 3, "MTA": 4}
METHODS = ("zero-shot", "shared-r16", "shared-r80", "task-routed")
DEFAULT_MODEL_BASELINES = {
    "qwen2-vl-2b": {
        "display_name": "Qwen2-VL-2B-Instruct",
        "backend": "hf-chat",
        "model": str(DEFAULT_BASE_MODEL),
    },
    "qwen2.5-vl-3b": {
        "display_name": "Qwen2.5-VL-3B-Instruct",
        "backend": "hf-chat",
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
    },
    "internvl2.5-2b": {
        "display_name": "InternVL2.5-2B",
        "backend": "internvl",
        "model": "OpenGVLab/InternVL2_5-2B",
    },
    "minicpm-v-2.6": {
        "display_name": "MiniCPM-V-2.6",
        "backend": "minicpm",
        "model": "openbmb/MiniCPM-V-2_6",
    },
    "llava-1.5-7b": {
        "display_name": "LLaVA-1.5-7B",
        "backend": "hf-chat",
        "model": "llava-hf/llava-1.5-7b-hf",
    },
}
METRIC_COLUMNS = ("SEP", "DHO_Base", "DHO_Corner", "RCE", "ADR", "MTA", "CCC", "Macro_Avg")
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


@dataclass(frozen=True)
class SceneRecord:
    branch: str
    task_type: str
    group_id: str
    name: str
    image_path: str
    label_path: str
    qa_pairs: list[dict[str, Any]]


def stable_fraction(value: str, seed: int) -> float:
    digest = hashlib.sha256(f"{seed}:{value}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def assign_split(group_id: str, seed: int, train_ratio: float, val_ratio: float) -> str:
    value = stable_fraction(group_id, seed)
    if value < train_ratio:
        return "train"
    if value < train_ratio + val_ratio:
        return "val"
    return "test"


def source_group_id(branch: str, record: dict[str, Any], fallback: str) -> str:
    if record.get("original_bdd_name"):
        return f"BDD100K:{record['original_bdd_name']}"
    if record.get("original_name"):
        family = "DAIR-V2X" if branch == "MTA" else "CODA"
        return f"{family}:{record['original_name']}"
    source = record.get("metadata", {}).get("source", branch)
    return f"{source}:{fallback}"


def resolve_image(images_dir: Path, name: str) -> Path | None:
    for suffix in (".jpg", ".jpeg", ".png", ".JPG", ".PNG"):
        candidate = images_dir / f"{name}{suffix}"
        if candidate.exists():
            return candidate
    return None


def load_scene_records(
    dataset_root: Path,
    split: str | None,
    seed: int,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    max_scenes_per_branch: int = 0,
    split_unit: str = "source",
) -> list[SceneRecord]:
    records: list[SceneRecord] = []
    for branch, parts in BRANCHES.items():
        branch_dir = dataset_root.joinpath(*parts)
        labels_dir = branch_dir / "label"
        images_dir = branch_dir / "images"
        if not labels_dir.is_dir() or not images_dir.is_dir():
            print(f"[warning] missing branch directories: {branch_dir}", file=sys.stderr)
            continue

        branch_records: list[SceneRecord] = []

        def read_label(label_path: Path) -> SceneRecord | None:
            try:
                record = json.loads(label_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                print(f"[warning] failed to read {label_path}: {exc}", file=sys.stderr)
                return None
            qa_pairs = record.get("qa_pairs") or []
            if not qa_pairs:
                return None
            name = str(record.get("name") or label_path.stem)
            image_path = resolve_image(images_dir, name)
            if image_path is None:
                return None
            group_id = source_group_id(branch, record, name)
            split_key = group_id if split_unit == "source" else f"{branch}:{label_path.name}"
            record_split = assign_split(split_key, seed, train_ratio, val_ratio)
            if split is not None and record_split != split:
                return None
            task_type = str(qa_pairs[0].get("task_type") or record.get("task_type") or branch.split("_")[0])
            return SceneRecord(
                branch=branch,
                task_type=task_type,
                group_id=group_id,
                name=name,
                image_path=str(image_path),
                label_path=str(label_path),
                qa_pairs=qa_pairs,
            )

        label_paths = sorted(labels_dir.glob("*.json"))
        with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
            for loaded in pool.map(read_label, label_paths, chunksize=64):
                if loaded is not None:
                    branch_records.append(loaded)

        if max_scenes_per_branch > 0:
            branch_records = branch_records[:max_scenes_per_branch]
        records.extend(branch_records)
    return records


def summarize_records(records: Sequence[SceneRecord]) -> dict[str, Any]:
    by_branch = Counter(record.branch for record in records)
    qa_by_branch = Counter()
    categories = defaultdict(Counter)
    unique_groups = defaultdict(set)
    for record in records:
        qa_by_branch[record.branch] += len(record.qa_pairs)
        unique_groups[record.branch].add(record.group_id)
        for qa in record.qa_pairs:
            categories[record.branch][str(qa.get("category", "unknown"))] += 1
    return {
        "scenes": len(records),
        "qa_pairs": sum(qa_by_branch.values()),
        "by_branch": {
            branch: {
                "scenes": by_branch[branch],
                "qa_pairs": qa_by_branch[branch],
                "unique_source_groups": len(unique_groups[branch]),
                "categories": dict(categories[branch]),
            }
            for branch in BRANCHES
        },
    }


def limit_by_branch(records: Sequence[SceneRecord], limit: int) -> list[SceneRecord]:
    if limit <= 0:
        return list(records)
    counts: Counter[str] = Counter()
    selected: list[SceneRecord] = []
    for record in records:
        if counts[record.branch] >= limit:
            continue
        selected.append(record)
        counts[record.branch] += 1
    return selected


def print_summary(summary: dict[str, Any]) -> None:
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def ensure_custom_peft_path() -> None:
    local_src = Path(__file__).resolve().parent / "peft_egg" / "src"
    if local_src.is_dir() and str(local_src) not in sys.path:
        sys.path.insert(0, str(local_src))


def build_instruction() -> str:
    return (
        "You are an expert autonomous driving cognitive coordinator.\n"
        "Select the single best option from the letter-coded choices.\n"
        "Output ONLY one capital letter (A, B, C, or D); do not add an explanation."
    )


def build_messages(question: str, answer: str | None = None) -> list[dict[str, Any]]:
    marker = "from the following options:"
    marker_pos = question.lower().find(marker)
    options: list[str] = []
    stem = question.strip()
    if marker_pos >= 0:
        list_start = question.find("[", marker_pos)
        list_end = question.find("]", list_start)
        if list_start >= 0 and list_end >= 0:
            parsed = ast.literal_eval(question[list_start : list_end + 1])
            options = [str(item) for item in parsed]
            stem = question[:marker_pos].strip()
    option_lines = "\n".join(f"{chr(65 + index)}. {option}" for index, option in enumerate(options))
    formatted_question = f"{stem}\nOptions:\n{option_lines}" if options else question
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": f"{build_instruction()}\n\nQuestion: {formatted_question}\nAnswer:"},
            ],
        }
    ]
    if answer is not None:
        target = str(answer)
        if options and target in options:
            target = chr(65 + options.index(target))
        messages.append({"role": "assistant", "content": [{"type": "text", "text": target}]})
    return messages


def clean_answer(value: Any) -> str:
    return re.sub(r'[\*\n\r"\'\[\]]', "", str(value).strip()).lower()


def parse_mcq_answer(prediction: str, label: str, question: str) -> bool:
    marker_pos = question.lower().find("from the following options:")
    if marker_pos >= 0:
        list_start = question.find("[", marker_pos)
        list_end = question.find("]", list_start)
        if list_start >= 0 and list_end >= 0:
            options = [str(item) for item in ast.literal_eval(question[list_start : list_end + 1])]
            if label in options:
                match = re.search(r"(?<![A-Za-z])([A-H])(?![A-Za-z])", str(prediction).upper())
                if match:
                    return ord(match.group(1)) - 65 == options.index(label)
    pred = clean_answer(prediction)
    target = clean_answer(label)
    if pred == target:
        return True
    if target and target in pred:
        matches = re.findall(r"\[(.*?)\]", question)
        if matches:
            options = [clean_answer(item) for item in re.split(r"'\s*,\s*'", matches[-1].strip("'"))]
            if not any(option and option != target and option in pred for option in options):
                return True
    return False


def import_ml_stack():
    ensure_custom_peft_path()
    import torch
    from PIL import Image
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration

    try:
        from peft.tuners.lora import update_lora_mapping
    except ImportError:
        update_lora_mapping = None
    return {
        "torch": torch,
        "Image": Image,
        "LoraConfig": LoraConfig,
        "PeftModel": PeftModel,
        "get_peft_model": get_peft_model,
        "prepare_model_for_kbit_training": prepare_model_for_kbit_training,
        "AutoProcessor": AutoProcessor,
        "BitsAndBytesConfig": BitsAndBytesConfig,
        "Qwen2VLForConditionalGeneration": Qwen2VLForConditionalGeneration,
        "update_lora_mapping": update_lora_mapping,
    }


def quantization_kwargs(stack: dict[str, Any], mode: str) -> tuple[dict[str, Any], Any]:
    torch = stack["torch"]
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    kwargs: dict[str, Any] = {"torch_dtype": dtype, "device_map": "auto", "trust_remote_code": True}
    if mode == "4bit":
        kwargs["quantization_config"] = stack["BitsAndBytesConfig"](
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
    return kwargs, dtype


def method_lora_config(stack: dict[str, Any], method: str, expert_rank: int = 16):
    config_cls = stack["LoraConfig"]
    if method == "shared-r16":
        return config_cls(
            r=16,
            num_rank_per_block=16,
            lora_alpha=32,
            target_modules=TARGET_MODULES,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    if method == "shared-r80":
        return config_cls(
            r=80,
            num_rank_per_block=80,
            lora_alpha=160,
            target_modules=TARGET_MODULES,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )
    if method == "task-routed":
        if stack["update_lora_mapping"] is None:
            raise RuntimeError("task-routed training requires the custom PEFT build with update_lora_mapping")
        try:
            return config_cls(
                r=5 * expert_rank,
                lora_alpha=10 * expert_rank,
                target_modules=TARGET_MODULES,
                lora_dropout=0.05,
                bias="none",
                task_type="CAUSAL_LM",
                num_rank_per_block=expert_rank,
            )
        except TypeError as exc:
            raise RuntimeError("the active PEFT LoraConfig does not support num_rank_per_block") from exc
    raise ValueError(f"unsupported trainable method: {method}")


def model_input_device(model):
    for parameter in model.parameters():
        if parameter.device.type != "meta":
            return parameter.device
    return model.device


def training_samples(records: Sequence[SceneRecord]) -> list[dict[str, Any]]:
    samples: list[dict[str, Any]] = []
    for record in records:
        for qa in record.qa_pairs:
            task_type = str(qa.get("task_type") or record.task_type)
            if task_type not in TASK_TO_BLOCK:
                continue
            samples.append(
                {
                    "image_path": record.image_path,
                    "group_id": record.group_id,
                    "branch": record.branch,
                    "task_type": task_type,
                    "block_id": int(qa.get("block_id", TASK_TO_BLOCK[task_type])),
                    "category": str(qa.get("category", "unknown")),
                    "question": str(qa.get("question", "")),
                    "answer": str(qa.get("label", "")),
                }
            )
    return samples


def make_training_inputs(stack: dict[str, Any], processor, sample: dict[str, Any], device):
    image = stack["Image"].open(sample["image_path"]).convert("RGB")
    prompt_text = processor.apply_chat_template(build_messages(sample["question"]), tokenize=False, add_generation_prompt=True)
    full_text = processor.apply_chat_template(
        build_messages(sample["question"], sample["answer"]), tokenize=False, add_generation_prompt=False
    )
    inputs = processor(text=[full_text], images=[image], padding=True, return_tensors="pt").to(device)
    prompt_inputs = processor(text=[prompt_text], images=[image], padding=True, return_tensors="pt").to(device)
    labels = inputs["input_ids"].clone()
    labels[:, : prompt_inputs["input_ids"].shape[1]] = -100
    pad_id = processor.tokenizer.pad_token_id
    if pad_id is not None:
        labels[labels == pad_id] = -100
    inputs["labels"] = labels
    return inputs


def train_command(args: argparse.Namespace) -> None:
    if args.method == "zero-shot":
        raise SystemExit("zero-shot has no training stage")
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = load_scene_records(
        dataset_root,
        split="train",
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_scenes_per_branch=args.max_scenes_per_branch,
        split_unit=args.split_unit,
    )
    samples = training_samples(records)
    if not samples:
        raise SystemExit("no training samples found; verify dataset branch directories")

    stack = import_ml_stack()
    torch = stack["torch"]
    processor = stack["AutoProcessor"].from_pretrained(
        args.base_model,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        trust_remote_code=True,
    )
    load_kwargs, _ = quantization_kwargs(stack, args.quantization)
    model = stack["Qwen2VLForConditionalGeneration"].from_pretrained(args.base_model, **load_kwargs)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    if args.quantization == "4bit":
        model = stack["prepare_model_for_kbit_training"](model)
    model = stack["get_peft_model"](model, method_lora_config(stack, args.method, args.expert_rank))
    model.train()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    device = model_input_device(model)
    global_step = 0
    optimizer.zero_grad(set_to_none=True)
    rng = random.Random(args.seed)
    loss_log: list[dict[str, Any]] = []

    for epoch in range(args.epochs):
        order = list(range(len(samples)))
        rng.shuffle(order)
        accumulated = 0
        for position, sample_index in enumerate(order, start=1):
            sample = samples[sample_index]
            if args.method == "task-routed":
                stack["update_lora_mapping"]([sample["block_id"]])
            try:
                inputs = make_training_inputs(stack, processor, sample, device)
                loss = model(**inputs).loss / args.gradient_accumulation
                loss.backward()
                accumulated += 1
            except torch.cuda.OutOfMemoryError:
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.empty_cache()
                print(f"[oom] skipped {sample['image_path']}", file=sys.stderr)
                continue

            should_step = accumulated >= args.gradient_accumulation or position == len(order)
            if should_step:
                torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                loss_log.append(
                    {
                        "epoch": epoch + 1,
                        "global_step": global_step,
                        "loss": float(loss.detach().cpu()) * args.gradient_accumulation,
                        "task_type": sample["task_type"],
                    }
                )
                accumulated = 0
                if global_step % args.log_every == 0:
                    print(f"epoch={epoch + 1} step={global_step} loss={loss_log[-1]['loss']:.6f}")

        checkpoint = output_dir / f"checkpoint_epoch_{epoch + 1}"
        model.save_pretrained(checkpoint)
        processor.save_pretrained(checkpoint)

    model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)
    (output_dir / "loss_log.json").write_text(json.dumps(loss_log, indent=2), encoding="utf-8")
    config = {key: value for key, value in vars(args).items() if key != "func"}
    config["dataset_summary"] = summarize_records(records)
    config["expanded_training_samples"] = len(samples)
    (output_dir / "experiment_config.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"saved adapter: {output_dir}")


def load_eval_model(stack: dict[str, Any], args: argparse.Namespace, method: str, adapter_path: Path | None):
    load_kwargs, _ = quantization_kwargs(stack, args.quantization)
    model = stack["Qwen2VLForConditionalGeneration"].from_pretrained(args.base_model, **load_kwargs)
    if adapter_path is not None:
        model = stack["PeftModel"].from_pretrained(model, adapter_path)
    model.eval()
    processor_source = adapter_path if adapter_path and (adapter_path / "preprocessor_config.json").exists() else Path(args.base_model)
    processor = stack["AutoProcessor"].from_pretrained(
        processor_source,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
        trust_remote_code=True,
    )
    if method == "task-routed" and stack["update_lora_mapping"] is None:
        raise RuntimeError("task-routed evaluation requires the custom PEFT build")
    return model, processor


def generate_answer(stack: dict[str, Any], model, processor, record: SceneRecord, qa: dict[str, Any], method: str, args):
    torch = stack["torch"]
    if method == "task-routed":
        task_type = str(qa.get("task_type") or record.task_type)
        block_id = int(qa.get("block_id", TASK_TO_BLOCK[task_type]))
        stack["update_lora_mapping"]([block_id])
    image = stack["Image"].open(record.image_path).convert("RGB")
    prompt = processor.apply_chat_template(build_messages(str(qa["question"])), tokenize=False, add_generation_prompt=True)
    device = model_input_device(model)
    inputs = processor(text=[prompt], images=[image], padding=True, return_tensors="pt").to(device)
    start = time.perf_counter()
    with torch.no_grad():
        generated = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            top_k=None,
        )
    latency = time.perf_counter() - start
    trimmed = [output[len(input_ids) :] for input_ids, output in zip(inputs["input_ids"], generated)]
    answer = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
    return answer, latency


def load_model_registry(path: str | None) -> dict[str, dict[str, str]]:
    registry = DEFAULT_MODEL_BASELINES if not path else json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(registry, dict) or not registry:
        raise SystemExit("model registry must be a non-empty JSON object")
    allowed_backends = {"hf-chat", "internvl", "minicpm"}
    normalized: dict[str, dict[str, str]] = {}
    for key, value in registry.items():
        if not isinstance(value, dict) or not value.get("model") or value.get("backend") not in allowed_backends:
            raise SystemExit(f"invalid model registry entry: {key}")
        normalized[str(key)] = {
            "display_name": str(value.get("display_name") or key),
            "backend": str(value["backend"]),
            "model": str(value["model"]),
        }
    return normalized


def selected_model_specs(args: argparse.Namespace) -> list[tuple[str, dict[str, str]]]:
    registry = load_model_registry(args.models_config)
    selected = args.model or list(registry)
    unknown = [key for key in selected if key not in registry]
    if unknown:
        raise SystemExit(f"unknown model key(s): {', '.join(unknown)}")
    return [(key, registry[key]) for key in selected]


def local_model_missing(model_ref: str) -> bool:
    return bool(re.match(r"^[A-Za-z]:[\\/]", model_ref) or model_ref.startswith(("/", "./", "../"))) and not Path(model_ref).exists()


def internvl_pixels(stack: dict[str, Any], image, dtype, device, image_size: int = 448, max_tiles: int = 12):
    try:
        import torchvision.transforms as transforms
        from torchvision.transforms.functional import InterpolationMode
    except ImportError as exc:
        raise RuntimeError("InternVL backend requires torchvision") from exc

    width, height = image.size
    aspect = width / height
    ratios = sorted(
        {(i, j) for n in range(1, max_tiles + 1) for i in range(1, n + 1) for j in range(1, n + 1) if 1 <= i * j <= max_tiles},
        key=lambda pair: pair[0] * pair[1],
    )
    ratio = min(ratios, key=lambda pair: (abs(aspect - pair[0] / pair[1]), -(pair[0] * pair[1])))
    target_width, target_height = image_size * ratio[0], image_size * ratio[1]
    resized = image.resize((target_width, target_height))
    tiles = []
    for index in range(ratio[0] * ratio[1]):
        left = (index % ratio[0]) * image_size
        top = (index // ratio[0]) * image_size
        tiles.append(resized.crop((left, top, left + image_size, top + image_size)))
    if len(tiles) > 1:
        tiles.append(image.resize((image_size, image_size)))
    transform = transforms.Compose(
        [
            transforms.Lambda(lambda item: item.convert("RGB")),
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ]
    )
    return stack["torch"].stack([transform(tile) for tile in tiles]).to(dtype=dtype, device=device)


def load_external_runtime(stack: dict[str, Any], spec: dict[str, str], args: argparse.Namespace) -> dict[str, Any]:
    from transformers import AutoModel, AutoTokenizer

    model_ref = spec["model"]
    backend = spec["backend"]
    load_kwargs, dtype = quantization_kwargs(stack, args.quantization)
    if backend == "hf-chat":
        from transformers import AutoProcessor
        try:
            from transformers import AutoModelForImageTextToText as AutoVisionModel
        except ImportError:
            try:
                from transformers import AutoModelForVision2Seq as AutoVisionModel
            except ImportError:
                AutoVisionModel = AutoModel
        model = AutoVisionModel.from_pretrained(model_ref, **load_kwargs).eval()
        processor = AutoProcessor.from_pretrained(
            model_ref, min_pixels=args.min_pixels, max_pixels=args.max_pixels, trust_remote_code=True
        )
        return {"backend": backend, "model": model, "processor": processor, "dtype": dtype}
    if backend == "internvl":
        model = AutoModel.from_pretrained(model_ref, **load_kwargs).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True, use_fast=False)
        return {"backend": backend, "model": model, "tokenizer": tokenizer, "dtype": dtype}
    if backend == "minicpm":
        model = AutoModel.from_pretrained(model_ref, **load_kwargs).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_ref, trust_remote_code=True)
        return {"backend": backend, "model": model, "tokenizer": tokenizer, "dtype": dtype}
    raise ValueError(f"unsupported backend: {backend}")


def generate_external_answer(stack: dict[str, Any], runtime: dict[str, Any], record: SceneRecord, qa: dict[str, Any], args):
    torch = stack["torch"]
    image = stack["Image"].open(record.image_path).convert("RGB")
    prompt_text = f"{build_instruction()}\n\nQuestion: {qa['question']}\nAnswer:"
    backend = runtime["backend"]
    model = runtime["model"]
    start = time.perf_counter()
    if backend == "hf-chat":
        processor = runtime["processor"]
        prompt = processor.apply_chat_template(build_messages(str(qa["question"])), tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[prompt], images=[image], padding=True, return_tensors="pt").to(model_input_device(model))
        with torch.no_grad():
            generated = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        trimmed = [output[len(input_ids) :] for input_ids, output in zip(inputs["input_ids"], generated)]
        answer = processor.batch_decode(trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    elif backend == "internvl":
        pixels = internvl_pixels(stack, image, runtime["dtype"], model_input_device(model), max_tiles=args.internvl_max_tiles)
        with torch.no_grad():
            answer = model.chat(
                runtime["tokenizer"], pixels, prompt_text,
                {"max_new_tokens": args.max_new_tokens, "do_sample": False},
            )
    elif backend == "minicpm":
        messages = [{"role": "user", "content": [image, prompt_text]}]
        with torch.no_grad():
            answer = model.chat(
                image=None, msgs=messages, tokenizer=runtime["tokenizer"],
                sampling=False, max_new_tokens=args.max_new_tokens,
            )
    else:
        raise ValueError(f"unsupported backend: {backend}")
    return str(answer).strip(), time.perf_counter() - start


def evaluate_records(method: str, records: Sequence[SceneRecord], generator, log_every: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    predictions: list[dict[str, Any]] = []
    for scene_index, record in enumerate(records, start=1):
        for qa in record.qa_pairs:
            prediction, latency = generator(record, qa)
            label = str(qa.get("label", ""))
            question = str(qa.get("question", ""))
            predictions.append(
                {
                    "method": method,
                    "branch": record.branch,
                    "group_id": record.group_id,
                    "image_name": record.name,
                    "image_path": record.image_path,
                    "task_type": str(qa.get("task_type", record.task_type)),
                    "category": str(qa.get("category", "unknown")),
                    "prediction": prediction,
                    "label": label,
                    "correct": parse_mcq_answer(prediction, label, question),
                    "latency_seconds": latency,
                }
            )
        if scene_index % log_every == 0:
            print(f"  {method}: {scene_index}/{len(records)} scenes")
    return predictions, compute_metrics(predictions)


def compute_metrics(predictions: Sequence[dict[str, Any]]) -> dict[str, Any]:
    scene_results: dict[tuple[str, str], dict[str, bool]] = defaultdict(dict)
    rce_by_group: dict[str, bool] = {}
    adr_by_group: dict[str, bool] = {}
    for item in predictions:
        scene_results[(item["branch"], item["group_id"])][item["category"]] = bool(item["correct"])
        if item["branch"] == "RCE" and item["category"] == "right_of_way":
            rce_by_group[item["group_id"]] = bool(item["correct"])
        if item["branch"] == "ADR" and item["category"] == "longitudinal_decision":
            adr_by_group[item["group_id"]] = bool(item["correct"])

    scores: dict[str, list[float]] = defaultdict(list)
    for (branch, _), categories in scene_results.items():
        if branch == "SEP":
            scores[branch].append(sum(categories.values()) / len(categories))
        elif branch in ("DHO_Base", "DHO_Corner"):
            scores[branch].append(float(bool(categories) and all(categories.values())))
        elif branch == "RCE":
            scores[branch].append(
                0.7 * float(categories.get("right_of_way", False))
                + 0.3 * float(categories.get("signal_constraint", False))
            )
        elif branch == "ADR":
            required = ("longitudinal_decision", "defensive_focus", "decision_rationale")
            scores[branch].append(float(all(categories.get(key, False) for key in required)))
        elif branch == "MTA":
            required = ("congestion_level", "static_anomaly", "vru_assessment", "v2x_synthesis")
            scores[branch].append(float(all(categories.get(key, False) for key in required)))

    metrics: dict[str, Any] = {}
    for branch in BRANCHES:
        values = scores.get(branch, [])
        metrics[branch] = sum(values) / len(values) if values else None
        metrics[f"N_{branch}"] = len(values)

    shared_groups = sorted(set(rce_by_group) & set(adr_by_group))
    denominator = sum(1 for key in shared_groups if rce_by_group[key])
    numerator = sum(1 for key in shared_groups if rce_by_group[key] and adr_by_group[key])
    metrics["CCC"] = numerator / denominator if denominator else None
    metrics["CCC_numerator"] = numerator
    metrics["CCC_denominator"] = denominator
    branch_values = [metrics[branch] for branch in BRANCHES if metrics[branch] is not None]
    metrics["Macro_Avg"] = sum(branch_values) / len(branch_values) if branch_values else None
    metrics["qa_predictions"] = len(predictions)
    return metrics


def write_method_outputs(output_dir: Path, method: str, summary: dict[str, Any], predictions: Sequence[dict[str, Any]]) -> None:
    method_dir = output_dir / method
    method_dir.mkdir(parents=True, exist_ok=True)
    (method_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    if predictions:
        with (method_dir / "predictions.jsonl").open("w", encoding="utf-8") as handle:
            for item in predictions:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def blank_summary(method: str, status: str, adapter: str | None = None) -> dict[str, Any]:
    summary: dict[str, Any] = {"method": method, "status": status, "adapter": adapter, "metrics": {}}
    for metric in METRIC_COLUMNS:
        summary["metrics"][metric] = None
    return summary


def adapter_paths(args: argparse.Namespace) -> dict[str, Path | None]:
    return {
        "zero-shot": None,
        "shared-r16": Path(args.adapter_shared_r16) if args.adapter_shared_r16 else None,
        "shared-r80": Path(args.adapter_shared_r80) if args.adapter_shared_r80 else None,
        "task-routed": Path(args.adapter_task_routed) if args.adapter_task_routed else None,
    }


def write_comparison_table(output_dir: Path, summaries: Sequence[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    include_model_fields = any("display_name" in summary for summary in summaries)
    columns = ["method", "status"]
    if include_model_fields:
        columns.extend(["display_name", "backend", "model"])
    columns.extend(METRIC_COLUMNS)
    with (output_dir / "comparison.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for summary in summaries:
            row = {"method": summary["method"], "status": summary["status"]}
            if include_model_fields:
                row.update(
                    {
                        "display_name": summary.get("display_name", ""),
                        "backend": summary.get("backend", ""),
                        "model": summary.get("model", ""),
                    }
                )
            for metric in METRIC_COLUMNS:
                value = summary.get("metrics", {}).get(metric)
                row[metric] = "" if value is None else f"{value:.6f}"
            writer.writerow(row)
    (output_dir / "comparison.json").write_text(json.dumps(list(summaries), ensure_ascii=False, indent=2), encoding="utf-8")


def compare_command(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    paths = adapter_paths(args)
    if args.reserve_only:
        summaries = [blank_summary(method, "pending_execution", str(paths[method]) if paths[method] else None) for method in METHODS]
        for summary in summaries:
            write_method_outputs(output_dir, summary["method"], summary, [])
        write_comparison_table(output_dir, summaries)
        print(f"reserved result files: {output_dir}")
        return

    records = load_scene_records(
        Path(args.dataset_root),
        split="test",
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_scenes_per_branch=args.max_scenes_per_branch,
    )
    if not records:
        raise SystemExit("no test records found")
    stack = import_ml_stack()
    summaries: list[dict[str, Any]] = []

    for method in METHODS:
        adapter = paths[method]
        if method != "zero-shot" and (adapter is None or not adapter.is_dir()):
            summary = blank_summary(method, "pending_checkpoint", str(adapter) if adapter else None)
            write_method_outputs(output_dir, method, summary, [])
            summaries.append(summary)
            continue

        print(f"evaluating {method}")
        model, processor = load_eval_model(stack, args, method, adapter)
        predictions: list[dict[str, Any]] = []
        for scene_index, record in enumerate(records, start=1):
            for qa in record.qa_pairs:
                prediction, latency = generate_answer(stack, model, processor, record, qa, method, args)
                label = str(qa.get("label", ""))
                question = str(qa.get("question", ""))
                predictions.append(
                    {
                        "method": method,
                        "branch": record.branch,
                        "group_id": record.group_id,
                        "image_name": record.name,
                        "image_path": record.image_path,
                        "task_type": str(qa.get("task_type", record.task_type)),
                        "category": str(qa.get("category", "unknown")),
                        "prediction": prediction,
                        "label": label,
                        "correct": parse_mcq_answer(prediction, label, question),
                        "latency_seconds": latency,
                    }
                )
            if scene_index % args.log_every_scenes == 0:
                print(f"  {method}: {scene_index}/{len(records)} scenes")
        metrics = compute_metrics(predictions)
        summary = {
            "method": method,
            "status": "complete",
            "adapter": str(adapter) if adapter else None,
            "dataset_summary": summarize_records(records),
            "metrics": metrics,
        }
        write_method_outputs(output_dir, method, summary, predictions)
        summaries.append(summary)
        del model, processor
        gc.collect()
        if stack["torch"].cuda.is_available():
            stack["torch"].cuda.empty_cache()

    write_comparison_table(output_dir, summaries)
    print(f"comparison results: {output_dir}")


def compare_models_command(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    specs = selected_model_specs(args)
    method_rows = [(key, spec) for key, spec in specs] + [
        (
            "causalscene-ours",
            {
                "display_name": "CausalScene (Task-routed MoE-LoRA)",
                "backend": "task-routed",
                "model": str(args.base_model),
            },
        )
    ]
    if args.reserve_only:
        summaries = []
        for key, spec in method_rows:
            summary = blank_summary(key, "pending_execution", args.causalscene_adapter if key == "causalscene-ours" else None)
            summary.update({"display_name": spec["display_name"], "backend": spec["backend"], "model": spec["model"]})
            write_method_outputs(output_dir, key, summary, [])
            summaries.append(summary)
        write_comparison_table(output_dir, summaries)
        print(f"reserved cross-model result files: {output_dir}")
        return

    records = load_scene_records(
        Path(args.dataset_root), split="test", seed=args.seed,
        train_ratio=args.train_ratio, val_ratio=args.val_ratio,
        max_scenes_per_branch=args.max_scenes_per_branch,
    )
    if not records:
        raise SystemExit("no test records found")
    stack = import_ml_stack()
    summaries: list[dict[str, Any]] = []

    for key, spec in method_rows:
        print(f"evaluating {spec['display_name']}")
        if key == "causalscene-ours":
            adapter = Path(args.causalscene_adapter) if args.causalscene_adapter else None
            if adapter is None or not adapter.is_dir():
                summary = blank_summary(key, "pending_checkpoint", str(adapter) if adapter else None)
                summary.update({"display_name": spec["display_name"], "backend": spec["backend"], "model": spec["model"]})
                write_method_outputs(output_dir, key, summary, [])
                summaries.append(summary)
                continue
            model, processor = load_eval_model(stack, args, "task-routed", adapter)
            generator = lambda record, qa: generate_answer(stack, model, processor, record, qa, "task-routed", args)
            predictions, metrics = evaluate_records(key, records, generator, args.log_every_scenes)
            del generator, model, processor
        else:
            if local_model_missing(spec["model"]):
                summary = blank_summary(key, "pending_checkpoint")
                summary.update({"display_name": spec["display_name"], "backend": spec["backend"], "model": spec["model"]})
                write_method_outputs(output_dir, key, summary, [])
                summaries.append(summary)
                continue
            try:
                runtime = load_external_runtime(stack, spec, args)
                generator = lambda record, qa: generate_external_answer(stack, runtime, record, qa, args)
                predictions, metrics = evaluate_records(key, records, generator, args.log_every_scenes)
                del generator, runtime
            except Exception as exc:
                if not args.continue_on_error:
                    raise
                print(f"[warning] {key} failed: {exc}", file=sys.stderr)
                summary = blank_summary(key, "failed")
                summary.update({"display_name": spec["display_name"], "backend": spec["backend"], "model": spec["model"], "error": str(exc)})
                write_method_outputs(output_dir, key, summary, [])
                summaries.append(summary)
                continue

        summary = {
            "method": key,
            "display_name": spec["display_name"],
            "status": "complete",
            "backend": spec["backend"],
            "model": spec["model"],
            "dataset_summary": summarize_records(records),
            "metrics": metrics,
        }
        write_method_outputs(output_dir, key, summary, predictions)
        summaries.append(summary)
        gc.collect()
        if stack["torch"].cuda.is_available():
            stack["torch"].cuda.empty_cache()

    write_comparison_table(output_dir, summaries)
    print(f"cross-model comparison results: {output_dir}")


def inspect_command(args: argparse.Namespace) -> None:
    output: dict[str, Any] = {
        "dataset_root": args.dataset_root,
        "seed": args.seed,
        "split_policy": {"train": args.train_ratio, "val": args.val_ratio, "test": 1 - args.train_ratio - args.val_ratio},
        "splits": {},
    }
    all_records = load_scene_records(
        Path(args.dataset_root),
        split=None,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        max_scenes_per_branch=0,
    )
    for split in ("train", "val", "test"):
        records = [
            record
            for record in all_records
            if assign_split(record.group_id, args.seed, args.train_ratio, args.val_ratio) == split
        ]
        records = limit_by_branch(records, args.max_scenes_per_branch)
        output["splits"][split] = summarize_records(records)
    print_summary(output)
    if args.output:
        path = Path(args.output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")


def add_common_dataset_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--max-scenes-per-branch", type=int, default=0, help="0 means all scenes")
    parser.add_argument("--split-unit", choices=("source", "record"), default="source")


def add_model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--base-model", default=str(DEFAULT_BASE_MODEL))
    parser.add_argument("--quantization", choices=("bf16", "4bit"), default="4bit")
    parser.add_argument("--min-pixels", type=int, default=256 * 28 * 28)
    parser.add_argument("--max-pixels", type=int, default=768 * 28 * 28)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect", help="audit source-grouped train/val/test splits")
    add_common_dataset_args(inspect_parser)
    inspect_parser.add_argument("--output")
    inspect_parser.set_defaults(func=inspect_command)

    train_parser = subparsers.add_parser("train", help="train one comparison adapter")
    add_common_dataset_args(train_parser)
    add_model_args(train_parser)
    train_parser.add_argument("--method", choices=METHODS[1:], required=True)
    train_parser.add_argument("--expert-rank", type=int, default=16, help="active rank per routed expert")
    train_parser.add_argument("--output-dir", required=True)
    train_parser.add_argument("--epochs", type=int, default=2)
    train_parser.add_argument("--learning-rate", type=float, default=2e-4)
    train_parser.add_argument("--weight-decay", type=float, default=0.01)
    train_parser.add_argument("--gradient-accumulation", type=int, default=4)
    train_parser.add_argument("--max-grad-norm", type=float, default=1.0)
    train_parser.add_argument("--log-every", type=int, default=100)
    train_parser.set_defaults(func=train_command)

    compare_parser = subparsers.add_parser("compare", help="evaluate all methods and write CSV/JSON summaries")
    add_common_dataset_args(compare_parser)
    add_model_args(compare_parser)
    compare_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT))
    compare_parser.add_argument("--adapter-shared-r16")
    compare_parser.add_argument("--adapter-shared-r80")
    compare_parser.add_argument("--adapter-task-routed")
    compare_parser.add_argument("--max-new-tokens", type=int, default=32)
    compare_parser.add_argument("--log-every-scenes", type=int, default=20)
    compare_parser.add_argument("--reserve-only", action="store_true", help="write blank result files without inference")
    compare_parser.set_defaults(func=compare_command)

    models_parser = subparsers.add_parser(
        "compare-models", help="evaluate CausalScene against zero-shot baselines from other VLM families"
    )
    add_common_dataset_args(models_parser)
    add_model_args(models_parser)
    models_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_ROOT / "cross_model"))
    models_parser.add_argument(
        "--model", action="append",
        help="registry key; repeat to select multiple models (default: all registry entries)",
    )
    models_parser.add_argument("--models-config", help="optional JSON registry replacing the built-in baseline registry")
    models_parser.add_argument("--causalscene-adapter", help="task-routed adapter checkpoint for the proposed method")
    models_parser.add_argument("--max-new-tokens", type=int, default=32)
    models_parser.add_argument("--log-every-scenes", type=int, default=20)
    models_parser.add_argument("--internvl-max-tiles", type=int, default=12)
    models_parser.add_argument("--continue-on-error", action="store_true")
    models_parser.add_argument("--reserve-only", action="store_true", help="write blank result files without inference")
    models_parser.set_defaults(func=compare_models_command)
    return parser


def validate_ratios(args: argparse.Namespace) -> None:
    if hasattr(args, "train_ratio"):
        if args.train_ratio <= 0 or args.val_ratio < 0 or args.train_ratio + args.val_ratio >= 1:
            raise SystemExit("train_ratio and val_ratio must leave a positive test split")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_ratios(args)
    args.func(args)


if __name__ == "__main__":
    main()
