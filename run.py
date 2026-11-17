import argparse
import json
import os
import torch
from PIL import Image


DEEPSEEK_MODEL_ID = os.environ.get("TRAFFIC_DEEPSEEK_MODEL", "deepseek-ai/deepseek-vl2-tiny")
LLAVA_MODEL_ID = os.environ.get("TRAFFIC_LLAVA_MODEL", "llava-hf/llava-1.5-7b-hf")
QWEN_MODEL_ID = os.environ.get("TRAFFIC_QWEN_MODEL", "Qwen/Qwen2-VL-2B-Instruct")
PHI35_MODEL_ID = os.environ.get("TRAFFIC_PHI_MODEL", "microsoft/Phi-3.5-vision-instruct")

def build_single_prompt(question):
    return (
        "You are an expert in autonomous driving scene analysis. I will provide you with an image and a question.\n"
        "You must answer the question strictly using ONLY the provided labels or 'yes'/'no'. "
        "Do not include any explanations, prefixes, or conversational text. Just output the exact label.\n\n"
        f"Question: {question}\nAnswer:"
    )


def calculate_score(predictions, ground_truths):
    if len(predictions) != len(ground_truths):
        return [0] * len(ground_truths)

    scores = []
    for pred, truth in zip(predictions, ground_truths):
        pred_clean = str(pred).lower().strip()
        truth_clean = str(truth).lower().strip()

        if ',' in truth_clean:
            truth_set = {x.strip() for x in truth_clean.split(',')}
            pred_set = {x.strip() for x in pred_clean.split(',')}
            scores.append(1 if truth_set == pred_set else 0)
        else:
            scores.append(1 if pred_clean == truth_clean else 0)
    return scores


def call_local_deepseek_vl(image_path, questions):
    print(f"\nLoading DeepSeek-VL2 ({DEEPSEEK_MODEL_ID})...")
    from deepseek_vl2.models import DeepseekVLV2Processor, DeepseekVLV2ForCausalLM

    vl_chat_processor = DeepseekVLV2Processor.from_pretrained(DEEPSEEK_MODEL_ID)
    vl_gpt = DeepseekVLV2ForCausalLM.from_pretrained(
        DEEPSEEK_MODEL_ID, torch_dtype=torch.bfloat16
    ).to("cuda").eval()

    pil_image = Image.open(image_path).convert("RGB")
    predictions = []

    print("Model loaded. Running inference for each question...")
    for idx, question in enumerate(questions):
        prompt = build_single_prompt(question)
        conversation = [
            {"role": "<|User|>", "content": f"<image>\n{prompt}"},
            {"role": "<|Assistant|>", "content": ""}
        ]

        prepare_inputs = vl_chat_processor(
            conversations=conversation, images=[pil_image], force_batchify=True, system_prompt=""
        ).to(vl_gpt.device, dtype=torch.bfloat16)

        with torch.no_grad():
            inputs_embeds = vl_gpt.prepare_inputs_embeds(**prepare_inputs)
            outputs = vl_gpt.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=prepare_inputs.attention_mask,
                pad_token_id=vl_chat_processor.tokenizer.eos_token_id,
                bos_token_id=vl_chat_processor.tokenizer.bos_token_id,
                eos_token_id=vl_chat_processor.tokenizer.eos_token_id,
                max_new_tokens=32,
                do_sample=False,
                use_cache=True
            )
            generated_text = vl_chat_processor.tokenizer.decode(
                outputs[0].cpu().tolist(), skip_special_tokens=True
            )

        answer = generated_text.strip()
        print(f"Answer {idx + 1}/{len(questions)}: {answer}")
        predictions.append(answer)

    del vl_gpt
    del vl_chat_processor
    torch.cuda.empty_cache()
    return predictions


def call_local_llava(image_path, questions):
    print(f"\nLoading LLaVA ({LLAVA_MODEL_ID})...")
    from transformers import AutoTokenizer, AutoImageProcessor, LlavaProcessor, LlavaForConditionalGeneration

    tokenizer = AutoTokenizer.from_pretrained(LLAVA_MODEL_ID, use_fast=False)
    image_processor = AutoImageProcessor.from_pretrained(LLAVA_MODEL_ID)
    processor = LlavaProcessor(image_processor=image_processor, tokenizer=tokenizer)

    model = LlavaForConditionalGeneration.from_pretrained(
        LLAVA_MODEL_ID,
        torch_dtype=torch.float16
    ).to("cuda").eval()

    pil_image = Image.open(image_path).convert("RGB")
    predictions = []

    print("Model loaded. Running inference for each question...")
    for idx, question in enumerate(questions):
        raw_prompt = build_single_prompt(question)
        prompt = f"USER: <image>\n{raw_prompt} ASSISTANT:"
        inputs = processor(text=prompt, images=pil_image, return_tensors="pt").to("cuda", torch.float16)

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=32, do_sample=False)
            generated_text = processor.decode(outputs[0], skip_special_tokens=True)

        if "ASSISTANT:" in generated_text:
            answer = generated_text.split("ASSISTANT:")[-1].strip()
        else:
            answer = generated_text.strip()

        print(f"Answer {idx + 1}/{len(questions)}: {answer}")
        predictions.append(answer)

    del model
    del processor
    torch.cuda.empty_cache()
    return predictions


def call_local_qwen(image_path, questions):
    print(f"\nLoading Qwen2-VL-2B ({QWEN_MODEL_ID})...")
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration

    processor = AutoProcessor.from_pretrained(QWEN_MODEL_ID)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        QWEN_MODEL_ID,
        torch_dtype=torch.float16
    ).to("cuda").eval()

    pil_image = Image.open(image_path).convert("RGB")
    predictions = []

    print("Model loaded. Running inference for each question...")
    for idx, question in enumerate(questions):
        raw_prompt = build_single_prompt(question)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": raw_prompt}
                ]
            }
        ]

        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(
            text=[text],
            images=pil_image,
            padding=True,
            return_tensors="pt"
        ).to("cuda")

        for k, v in inputs.items():
            if v.dtype == torch.float32:
                inputs[k] = v.to(torch.float16)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        generated_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        answer = generated_text.strip()
        print(f"Answer {idx + 1}/{len(questions)}: {answer}")
        predictions.append(answer)

    del model
    del processor
    torch.cuda.empty_cache()
    return predictions


def call_local_phi35_vision(image_path, questions):

    print(f"\nLoading Phi-3.5-vision-instruct ({PHI35_MODEL_ID})...")
    from transformers import AutoModelForCausalLM, AutoProcessor

    processor = AutoProcessor.from_pretrained(PHI35_MODEL_ID, trust_remote_code=True)


    model = AutoModelForCausalLM.from_pretrained(
        PHI35_MODEL_ID,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        _attn_implementation="eager"
    ).to("cuda").eval()

    pil_image = Image.open(image_path).convert("RGB")
    predictions = []

    print("Model loaded. Running inference for each question...")
    for idx, question in enumerate(questions):
        raw_prompt = build_single_prompt(question)


        prompt = f"<|user|>\n<|image_1|>\n{raw_prompt}<|end|>\n<|assistant|>\n"

        inputs = processor(prompt, [pil_image], return_tensors="pt").to("cuda")


        for k, v in inputs.items():
            if v.dtype == torch.float32:
                inputs[k] = v.to(torch.float16)

        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=32,
                do_sample=False,
                eos_token_id=processor.tokenizer.eos_token_id
            )


        generated_ids_trimmed = generated_ids[0][inputs['input_ids'].shape[1]:]

        answer = processor.tokenizer.decode(
            generated_ids_trimmed,
            skip_special_tokens=True
        ).strip()

        print(f"Answer {idx + 1}/{len(questions)}: {answer}")
        predictions.append(answer)

    del model
    del processor
    torch.cuda.empty_cache()
    return predictions


def main():
    model_callers = {
        "deepseek": call_local_deepseek_vl,
        "llava": call_local_llava,
        "qwen": call_local_qwen,
        "phi": call_local_phi35_vision,
    }
    parser = argparse.ArgumentParser(description="Evaluate vision-language models on one image.")
    parser.add_argument("--image", required=True, help="Path to the input image")
    parser.add_argument("--qa-file", required=True, help="JSON file with questions and labels arrays")
    parser.add_argument("--model", action="append", choices=model_callers, help="Repeat to select multiple models")
    args = parser.parse_args()

    with open(args.qa_file, encoding="utf-8") as handle:
        sample = json.load(handle)
    if not isinstance(sample, dict):
        parser.error("the QA file must contain a JSON object")
    questions = sample.get("questions")
    ground_truth = sample.get("labels")
    if not isinstance(questions, list) or not isinstance(ground_truth, list) or not questions:
        parser.error("the QA file must contain nonempty questions and labels arrays")
    if len(questions) != len(ground_truth):
        parser.error("the QA file must contain one label per question")

    results_summary = {}
    for model_name in args.model or ["deepseek", "phi"]:
        caller_func = model_callers[model_name]
        try:
            predictions = caller_func(args.image, questions)

            if not predictions:
                continue

            print(f"\n[{model_name}] Predictions: {predictions}")

            scores = calculate_score(predictions, ground_truth)
            total_score = sum(scores)

            results_summary[model_name] = {
                "predictions": predictions,
                "scores": scores,
                "accuracy": f"{total_score}/{len(ground_truth)}"
            }
            print(f"[{model_name}] Score: {total_score}/{len(ground_truth)}\n")

        except Exception as e:
            import traceback
            print(f"[{model_name}] Evaluation failed:\n")
            traceback.print_exc()

    print("-" * 40)
    print("Evaluation summary")
    print(json.dumps(results_summary, indent=4, ensure_ascii=False))


if __name__ == "__main__":
    main()
