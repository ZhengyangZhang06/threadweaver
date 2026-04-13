# Using `train.sh` for ThreadWeaver SFT

This guide explains how to run supervised fine-tuning (SFT) with `train.sh`.

## 1. Run from the correct directory

`train.sh` uses relative paths like `src/sft_threadweaver.py` and `configs/deepspeed_zero3_offload.json`, so run it from:

```bash
cd threadweaver/threadweaver_sft
```

## 2. Minimal command

```bash
./train.sh
```

By default, this uses:
- Base model: `BASE_MODEL` env var or `/storage/openpis/models/Qwen__Qwen3-8B`
- Dataset path: `TRAIN_DATA` env var or `./data/mult-10k-par`
- Output dir: `OUTPUT_DIR` env var or `ckpts/Q3-8B-131072-SFT-<timestamp>`

## 3. Common usage patterns

### Set model, data, and output explicitly

```bash
./train.sh \
  --original_model_path /path/to/Qwen3-8B-131072 \
  --dataset_dir /path/to/train.parquet \
  --output_dir ckpts/my-sft-run
```

### Equivalent aliases

- `--original_model_path`, `--base_model`, `--model_name` all set the base model path.
- `--dataset_dir`, `--dataset_path`, `--train_data` all set the training dataset path.

### Use env vars instead of CLI flags

```bash
BASE_MODEL=/path/to/model \
TRAIN_DATA=/path/to/train.parquet \
OUTPUT_DIR=ckpts/my-sft-run \
./train.sh
```

## 4. Pass extra trainer arguments

Any unknown args are forwarded to `src/sft_threadweaver.py` (TRL/HF trainer args).

Example:

```bash
./train.sh \
  --dataset_dir /path/to/train.parquet \
  --save_strategy steps \
  --save_steps 100 \
  --report_to none
```

Note:
- `train.sh` already sets many defaults (for example `--report_to wandb`, `--save_strategy no`).
- If you pass the same argument again, your later value is used.

## 5. What `train.sh` configures for you

`train.sh` launches:

```bash
torchrun --nproc-per-node gpu --master_port 12345 src/sft_threadweaver.py ...
```

Key built-in defaults include:
- `--template_name qwen`
- `--dataset_text_field qwen_text`
- `--block_size 40960`
- `--deepspeed configs/deepspeed_zero3_offload.json`
- `--bf16 True`
- `--gradient_checkpointing True`
- `--use-liger True`
- `--attn_implementation flex_attention`

Training hyperparameters currently set in the script:
- `lr=1e-5`
- `epochs=8`
- `micro_batch_size=1`
- `gradient_accumulation_steps=2`
- `weight_decay=1e-4`

If you want to change these defaults globally, edit `train.sh`.

## 6. Dataset expectations

- Provide a dataset path accepted by `load_dataset(...)` in `sft_threadweaver.py`.
- The training text column is expected to be `qwen_text` (set by `--dataset_text_field qwen_text` in `train.sh`).

## 7. Multi-GPU notes

- `torchrun --nproc-per-node gpu` uses all visible GPUs.
- To limit GPUs, set `CUDA_VISIBLE_DEVICES` first:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./train.sh --dataset_dir /path/to/train.parquet
```

## 8. Output

Model artifacts are written to your `--output_dir` (or `OUTPUT_DIR` env var), and by default include model/tokenizer outputs saved by the trainer.

