#!/bin/bash

# MiniGPT 模型优化脚本
# 用法: bash optimize.sh [light|medium|aggressive|benchmark|chat]

set -e

MODEL_PATH="./my_model.pt"
VOCAB_PATH="./vocab.json"
OPTIMIZED_MODEL="./model_optimized.pt"

show_help() {
    echo "MiniGPT 模型优化脚本"
    echo ""
    echo "用法: bash optimize.sh [命令]"
    echo ""
    echo "可用命令:"
    echo "  light       轻量级优化 (仅量化)"
    echo "  medium      中级优化 (量化 + 剪枝)"
    echo "  aggressive  激进优化 (量化 + 剪枝)"
    echo "  benchmark   性能测试"
    echo "  chat        使用优化模型对话"
    echo "  help        显示此帮助信息"
    echo ""
    echo "示例:"
    echo "  bash optimize.sh medium"
    echo "  bash optimize.sh chat"
}

optimize_light() {
    echo "=== 轻量级优化: 仅量化 ==="
    python3 optimize.py \
        --model_path $MODEL_PATH \
        --vocab_path $VOCAB_PATH \
        --output_path $OPTIMIZED_MODEL \
        --level light
    echo ""
    echo "优化完成! 模型保存至: $OPTIMIZED_MODEL"
    echo "使用以下命令测试优化后的模型:"
    echo "  bash optimize.sh chat"
}

optimize_medium() {
    echo "=== 中级优化: 量化 + 剪枝 ==="
    python3 optimize.py \
        --model_path $MODEL_PATH \
        --vocab_path $VOCAB_PATH \
        --output_path $OPTIMIZED_MODEL \
        --level medium \
        --sparsity 0.3
    echo ""
    echo "优化完成! 模型保存至: $OPTIMIZED_MODEL"
    echo "使用以下命令测试优化后的模型:"
    echo "  bash optimize.sh chat"
}

optimize_aggressive() {
    echo "=== 激进优化: 量化 + 剪枝 ==="
    python3 optimize.py \
        --model_path $MODEL_PATH \
        --vocab_path $VOCAB_PATH \
        --output_path $OPTIMIZED_MODEL \
        --level aggressive \
        --sparsity 0.5
    echo ""
    echo "优化完成! 模型保存至: $OPTIMIZED_MODEL"
    echo "使用以下命令测试优化后的模型:"
    echo "  bash optimize.sh chat"
}

run_benchmark() {
    echo "=== 性能测试 ==="
    python3 benchmark.py \
        --model_path $MODEL_PATH \
        --vocab_path $VOCAB_PATH \
        --compare_levels
}

chat_optimized() {
    echo "=== 优化模型对话 ==="
    if [ ! -f "$OPTIMIZED_MODEL" ]; then
        echo "优化模型不存在，先进行中级优化..."
        optimize_medium
    fi

    echo "启动优化模型对话 (启用KV Cache)..."
    python3 chat.py \
        --model_path $OPTIMIZED_MODEL \
        --vocab_path $VOCAB_PATH \
        --use_cache \
        --temperature 1.2 \
        --top_k 30
}

case "${1:-help}" in
    light)
        optimize_light
        ;;
    medium)
        optimize_medium
        ;;
    aggressive)
        optimize_aggressive
        ;;
    benchmark)
        run_benchmark
        ;;
    chat)
        chat_optimized
        ;;
    help|*)
        show_help
        ;;
esac
