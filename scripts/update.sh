#!/bin/bash

# MiniGPT 重新训练脚本
# 用法: bash scripts/update.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

echo "=== MiniGPT 重新训练 ==="
echo "删除现有模型和词表..."

rm -f ./models/my_model.pt ./data/vocab.json

echo "开始训练..."
bash scripts/train.sh

echo ""
echo "启动对话..."
bash scripts/chat.sh
