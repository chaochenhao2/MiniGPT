"""
模型优化整合模块 - 支持多种优化级别
用法：python optimize.py --model_path ./my_model.pt --vocab_path ./vocab.json --level medium
"""

import argparse
import torch
from train_llm import MiniGPT, CharTokenizer
from quantize import quantize_model, get_model_size, dynamic_quantization
from prune import prune_model, unstructured_global_pruning, structured_pruning, structured_pruning_heads, remove_pruning_reparametrization


def load_model(model_path, device='cpu'):
    checkpoint = torch.load(model_path, map_location=device)
    model_args = checkpoint['model_args']
    model = MiniGPT(**model_args)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, model_args


def optimize_light(model_path, vocab_path, output_path):
    print("=== 轻量级优化: 仅量化 ===")
    model, model_args = load_model(model_path)
    print(f"原始模型大小: {get_model_size(model):.2f} MB")

    quantized_model = dynamic_quantization(model)
    print(f"量化后模型大小: {get_model_size(quantized_model):.2f} MB")

    checkpoint = {
        'model_state_dict': quantized_model.state_dict(),
        'model_args': model_args,
        'quantized': True,
        'optimization_level': 'light'
    }
    torch.save(checkpoint, output_path)
    print(f"优化模型已保存至: {output_path}")
    return quantized_model


def optimize_medium(model_path, vocab_path, output_path, sparsity=0.3):
    print("=== 中级优化: 量化 + 剪枝 ===")
    model, model_args = load_model(model_path)
    print(f"原始模型大小: {get_model_size(model):.2f} MB")

    print("\n[1/2] 执行结构化剪枝...")
    pruned_model = structured_pruning(model, amount=sparsity, method='l1')
    pruned_model = structured_pruning_heads(pruned_model, heads_to_prune=2)
    pruned_model = remove_pruning_reparametrization(pruned_model)
    print(f"剪枝后模型大小: {get_model_size(pruned_model):.2f} MB")

    print("\n[2/2] 执行量化...")
    quantized_model = dynamic_quantization(pruned_model)
    print(f"量化后模型大小: {get_model_size(quantized_model):.2f} MB")

    checkpoint = {
        'model_state_dict': quantized_model.state_dict(),
        'model_args': model_args,
        'quantized': True,
        'pruned': True,
        'optimization_level': 'medium'
    }
    torch.save(checkpoint, output_path)
    print(f"优化模型已保存至: {output_path}")
    return quantized_model


def optimize_aggressive(model_path, vocab_path, output_path, sparsity=0.5):
    print("=== 激进优化: 量化 + 剪枝 + 推理优化提示 ===")
    model, model_args = load_model(model_path)
    print(f"原始模型大小: {get_model_size(model):.2f} MB")

    print("\n[1/3] 执行非结构化全局剪枝...")
    pruned_model = unstructured_global_pruning(model, amount=sparsity, method='l1')
    pruned_model = structured_pruning_heads(pruned_model, heads_to_prune=3)
    pruned_model = remove_pruning_reparametrization(pruned_model)
    print(f"剪枝后模型大小: {get_model_size(pruned_model):.2f} MB")

    print("\n[2/3] 执行量化...")
    quantized_model = dynamic_quantization(pruned_model)
    print(f"量化后模型大小: {get_model_size(quantized_model):.2f} MB")

    print("\n[3/3] 推理优化提示:")
    print("  - 使用 --use_cache 参数启用 KV Cache")
    print("  - 使用 --use_compile 参数启用 torch.compile")

    checkpoint = {
        'model_state_dict': quantized_model.state_dict(),
        'model_args': model_args,
        'quantized': True,
        'pruned': True,
        'optimization_level': 'aggressive'
    }
    torch.save(checkpoint, output_path)
    print(f"\n优化模型已保存至: {output_path}")
    return quantized_model


def main(args):
    if args.level == 'light':
        optimize_light(args.model_path, args.vocab_path, args.output_path)
    elif args.level == 'medium':
        optimize_medium(args.model_path, args.vocab_path, args.output_path, args.sparsity)
    elif args.level == 'aggressive':
        optimize_aggressive(args.model_path, args.vocab_path, args.output_path, args.sparsity)
    else:
        print(f"未知优化级别: {args.level}")
        return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='模型优化整合')
    parser.add_argument('--model_path', type=str, required=True, help='原始模型路径')
    parser.add_argument('--vocab_path', type=str, required=True, help='词表路径')
    parser.add_argument('--output_path', type=str, default='./model_optimized.pt', help='优化模型保存路径')
    parser.add_argument('--level', type=str, default='medium', choices=['light', 'medium', 'aggressive'],
                        help='优化级别: light(仅量化), medium(量化+剪枝), aggressive(全部)')
    parser.add_argument('--sparsity', type=float, default=0.3, help='剪枝比例 (0.0-1.0)')
    args = parser.parse_args()

    main(args)
