#!/bin/bash

# MiniGPT 对话脚本
# 用法: bash scripts/chat.sh [optimized|original|compiled] [temperature] [top_k]

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

MODE=${1:-original}
TEMPERATURE=${2:-1.2}
TOP_K=${3:-30}

cd "$PROJECT_DIR"

echo "=== MiniGPT 对话 ==="

case "$MODE" in
    optimized|opt)
        echo "模式: 优化模型 (启用KV Cache)"
        if [ ! -f "./models/model_optimized.pt" ]; then
            echo "优化模型不存在，先进行中级优化..."
            bash scripts/optimize.sh medium
        fi
        python3 src/chat.py \
            --model_path ./models/model_optimized.pt \
            --vocab_path ./data/vocab.json \
            --use_cache \
            --temperature $TEMPERATURE \
            --top_k $TOP_K
        ;;
    compiled|compile)
        echo "模式: 编译优化 (torch.compile)"
        python3 src/chat.py \
            --model_path ./models/my_model.pt \
            --vocab_path ./data/vocab.json \
            --use_cache \
            --use_compile \
            --temperature $TEMPERATURE \
            --top_k $TOP_K
        ;;
    original|*)
        echo "模式: 原始模型"
        python3 src/chat.py \
            --model_path ./models/my_model.pt \
            --vocab_path ./data/vocab.json \
            --temperature $TEMPERATURE \
            --top_k $TOP_K
        ;;
esac
