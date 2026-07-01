#!/bin/bash

# MiniGPT 训练脚本
# 用法: bash train.sh [epochs] [batch_size] [seq_len] [lr]

set -e

EPOCHS=${1:-3}
BATCH_SIZE=${2:-8}
SEQ_LEN=${3:-32}
LR=${4:-0.001}

echo "=== MiniGPT 训练 ==="
echo "参数: epochs=$EPOCHS, batch_size=$BATCH_SIZE, seq_len=$SEQ_LEN, lr=$LR"

python3 train_llm.py \
  --data_path ./data.txt \
  --save_model ./my_model.pt \
  --save_vocab ./vocab.json \
  --seq_len $SEQ_LEN \
  --epochs $EPOCHS \
  --batch_size $BATCH_SIZE \
  --lr $LR

echo ""
echo "训练完成! 模型保存至: ./my_model.pt"
echo "使用以下命令测试模型:"
echo "  bash chat.sh"
