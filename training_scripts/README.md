# SDXL PTI + LoRA Training

This directory contains the project training entrypoints.

## Setup

```bash
bash scripts/setup_conda_env.sh
bash scripts/download_sdxl_base.sh
```

Both scripts can be redirected for a remote NAS install:

```bash
CONDA_ENV_PREFIX=/mnt/nas-new/valencia/sdxl-style-lora/.conda/sdxl-lora \
  bash scripts/setup_conda_env.sh

MODEL_DIR=/mnt/nas-new/valencia/sdxl-style-lora/models/sdxl-base-1.0 \
  bash scripts/download_sdxl_base.sh
```

## Formal single-style training

The default launcher assumes at least 3 GPUs with 24GB VRAM each.

```bash
STYLE=jojo DATA_DIR=data/jojo NUM_PROCESSES=3 \
  bash training_scripts/run_sdxl_pti_lora_single.sh
```

Expected dataset layout:

```text
data/jojo/
  001.png
  001.txt
  002.png
  002.txt
```

If a matching `.txt` file is absent, the launcher uses:

```text
<style_headshot>, anime portrait, close-up face, character headshot
```

The default route is:

- SDXL base fp16 from `models/sdxl-base-1.0`
- 1024 x 1024
- batch size 1 per GPU
- gradient accumulation 4
- 500 TI-only steps
- 2000 total steps
- LoRA rank 16
- UNet LoRA plus text-encoder LoRA

## Local smoke test

This only checks that the training path runs and saves artifacts.

```bash
bash training_scripts/run_sdxl_pti_lora_smoke.sh
```

It uses the existing `data/` directory, 256 x 256 images, 1 TI step and 1 LoRA
step. Do not use the smoke output as a quality signal.

## Outputs

Each training run writes:

```text
outputs/sdxl_lora/<style>/
  pytorch_lora_weights.safetensors
  learned_embeds.safetensors
  training_config.json
  training_metadata.json
  checkpoint-*/...
```

## Generate from a trained LoRA

```bash
.conda/sdxl-lora/bin/python scripts/generate_sdxl_lora.py \
  --lora-dir outputs/sdxl_lora/jojo \
  --output-dir outputs/sdxl_lora_eval/jojo \
  --prompt "<jojo_headshot>, anime portrait, close-up face, clean line art" \
  --seeds 7 42
```

## Merge style LoRAs

The default merge method is `concat`: it preserves the weighted sum of LoRA
updates by stacking ranks. Two rank-16 LoRAs become one rank-32 LoRA; five
rank-16 LoRAs become one rank-80 LoRA. By default, the learned trigger-token
embeddings are averaged into one new trigger token.

```bash
.conda/sdxl-lora/bin/python scripts/merge_sdxl_loras.py \
  --lora-dir outputs/sdxl_lora/jojo \
  --lora-dir outputs/sdxl_lora/bocchi \
  --weight 0.5 \
  --weight 0.5 \
  --output-dir outputs/sdxl_lora/merged_jojo_bocchi \
  --placeholder-token "<merged_headshot>"
```

Then generate from the merged adapter:

```bash
.conda/sdxl-lora/bin/python scripts/generate_sdxl_lora.py \
  --lora-dir outputs/sdxl_lora/merged_jojo_bocchi \
  --output-dir outputs/sdxl_lora_eval/merged_jojo_bocchi \
  --prompt "<merged_headshot>, anime portrait, close-up face, clean line art"
```

To keep each input LoRA trigger token while still merging the LoRA weights, use
`--embed-merge keep`:

```bash
.conda/sdxl-lora/bin/python scripts/merge_sdxl_loras.py \
  --lora-dir outputs/sdxl_lora/ghibli \
  --lora-dir outputs/sdxl_lora/persona_5 \
  --lora-dir outputs/sdxl_lora/eva_rei \
  --output-dir outputs/sdxl_lora/merged_multitoken \
  --method concat \
  --embed-merge keep
```

The merged adapter can then use the original tokens in prompts:

```bash
.conda/sdxl-lora/bin/python scripts/generate_sdxl_lora.py \
  --lora-dir outputs/sdxl_lora/merged_multitoken \
  --output-dir outputs/sdxl_lora_eval/merged_multitoken \
  --prompt "<ghibli_headshot>, anime portrait, close-up face, clean line art" \
  --prompt "<persona_5_headshot>, anime portrait, close-up face, clean line art" \
  --prompt "<eva_rei_headshot>, anime portrait, close-up face, clean line art"
```

For report experiments, start with pairwise merges and compare multiple
`--lora-scale` values such as `0.4`, `0.6`, and `0.8`. Merging many strong style
LoRAs at once often washes out identity or produces noisy style conflicts.
