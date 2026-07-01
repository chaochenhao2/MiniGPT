#!/bin/bash

# MiniGPT 追加训练脚本
# 用法: bash add.sh [optimize|no-optimize]

set -e

OPTIMIZE=${1:-optimize}

echo "=== MiniGPT 追加训练 ==="

if [ ! -f "./data_new.txt" ] || [ ! -s "./data_new.txt" ]; then
    echo "错误: data_new.txt 不存在或为空"
    echo "请先在 data_new.txt 中添加训练数据"
    exit 1
fi

echo "[1/3] 使用新数据追加训练..."
python3 train_llm.py \
  --data_path ./data_new.txt \
  --save_model ./my_model.pt \
  --save_vocab ./vocab.json \
  --vocab_path ./vocab.json \
  --epochs 2 \
  --batch_size 8 \
  --seq_len 32 \
  --lr 0.0001

echo ""
echo "[2/3] 合并数据..."
cat ./data_new.txt >> ./data.txt
> ./data_new.txt
echo "数据已合并到 data.txt，data_new.txt 已清空"

echo ""
if [ "$OPTIMIZE" = "optimize" ]; then
    echo "[3/3] 优化模型并启动对话..."
    bash optimize.sh medium
    echo ""
    echo "启动优化模型对话..."
    bash chat.sh optimized
else
    echo "[3/3] 启动对话..."
    bash chat.sh
fi
