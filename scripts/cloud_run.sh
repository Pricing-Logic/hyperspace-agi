#!/bin/bash
set -euo pipefail

echo "=== SETUP ==="
cd /root/hyperspace
python3 -m pip install --upgrade pip > /dev/null 2>&1
python3 -m pip install --upgrade -r requirements-cuda.txt 2>&1 | tail -3
echo "READY"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "=== PHASE 1: CLEAN TRAIN ==="
mapfile -t traces < <(find experiments/local_traces -maxdepth 1 -name "*.jsonl" | sort)
echo "Found ${#traces[@]} trace files"
python3 -u scripts/train_clean.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --output experiments/clean_lora_v1 \
  --rank 16 --lr 5e-5 --epochs 3 --batch-size 4 \
  --traces "${traces[@]}"

echo "=== PHASE 2: A/B TEST ==="
python3 -u scripts/ab_test.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --adapter experiments/clean_lora_v1/adapter \
  --problems 30

echo "=== ALL DONE ==="
