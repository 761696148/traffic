# Traffic

Train and evaluate vision-language models on CausalScene and BDD100K. The custom PEFT package in `peft_egg/` implements task-routed LoRA.

## Setup

Use Python 3.10–3.12. From the repository root, install the dependencies.

~~~sh
python -m pip install -r requirements.txt
~~~

Use matching PyTorch and torchvision builds for your CUDA environment. For 4-bit training, install `bitsandbytes` on a supported GPU environment. For the optional DeepSeek backend, follow the [DeepSeek-VL2 installation instructions](https://github.com/deepseek-ai/DeepSeek-VL2#installation) in a separate Python environment and select `--model deepseek`.

## CausalScene experiments

Each CausalScene branch uses `images/` and `label/` subdirectories. The loader supports `SEP`, `DHO/bdd`, `DHO/Corner_Cases`, `RCE`, `ADR`, and `MTA`.

~~~sh
python causalscene_experiments.py inspect --dataset-root data/CAUSALSCENE
python causalscene_experiments.py train --dataset-root data/CAUSALSCENE --method task-routed --output-dir outputs/task_routed
python causalscene_experiments.py compare --dataset-root data/CAUSALSCENE --adapter-task-routed outputs/task_routed
python causalscene_experiments.py compare-models --dataset-root data/CAUSALSCENE --model qwen2-vl-2b
~~~

Use `--help` on each command for options. `--base-model` accepts a Hugging Face repository ID or a model directory. The `compare-models` command also accepts a model registry via `--models-config`.

## Other scripts

- `run.py --image IMAGE --qa-file QA_JSON --model qwen` scores one image provided by the caller. The JSON contains parallel `questions` and `labels` arrays. Model choices are `qwen`, `phi`, `llava`, and `deepseek`.
- `download_model.py MODEL_ID --output-dir models/MODEL` downloads weights on demand. The Hugging Face Hub uses its standard authentication and cache settings.
- `train.py`, `qwen2.py`, `phi3.5.py`, and `eval_qwen2vl_dynamic_lora.py` take BDD100K image and label directories from `TRAFFIC_IMAGE_DIR` and `TRAFFIC_LABEL_DIR`.
- `train_qwen2vl_dynamic_lora.py` uses the same variables for flat CausalDrive image and label directories. `eval_qwen.py` reads the branched dataset from `TRAFFIC_CAUSALSCENE_ROOT`. Set `TRAFFIC_USE_LORA=1` to evaluate with the trained adapter.
- `TRAFFIC_SAVE_DIR` controls where the standalone trainers save adapters. `TRAFFIC_LORA_DIR` identifies an adapter checkpoint for the standalone evaluators.
- Override the model references with `TRAFFIC_QWEN_MODEL`, `TRAFFIC_PHI_MODEL`, `TRAFFIC_LLAVA_MODEL`, or `TRAFFIC_DEEPSEEK_MODEL`.
