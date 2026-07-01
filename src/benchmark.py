"""
性能测试模块 - 对比不同优化方案的效果
用法：python benchmark.py --model_path ./my_model.pt --vocab_path ./vocab.json
"""

import argparse
import time
import torch
import torch.nn as nn
from train_llm import MiniGPT, CharTokenizer, generate_square_subsequent_mask
from quantize import dynamic_quantization, get_model_size
from prune import structured_pruning, structured_pruning_heads, remove_pruning_reparametrization


def load_model(model_path, device='cpu'):
    checkpoint = torch.load(model_path, map_location=device)
    model_args = checkpoint['model_args']
    model = MiniGPT(**model_args)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, model_args


def benchmark_inference(model, tokenizer, test_prompts, device='cpu', num_runs=3):
    model.eval()
    results = []

    for prompt in test_prompts:
        tokens = tokenizer.encode(prompt)
        if not tokens:
            continue

        input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(1).to(device)
        times = []

        for _ in range(num_runs):
            with torch.no_grad():
                attn_mask = generate_square_subsequent_mask(input_ids.size(0), device)
                start_time = time.time()
                logits = model(input_ids, attn_mask)
                end_time = time.time()
                times.append(end_time - start_time)

        avg_time = sum(times) / len(times)
        results.append({
            'prompt': prompt,
            'avg_time': avg_time,
            'tokens_per_sec': len(tokens) / avg_time
        })

    return results


def benchmark_generation(model, tokenizer, test_prompts, device='cpu', max_len=50, num_runs=3):
    model.eval()
    results = []

    for prompt in test_prompts:
        tokens = tokenizer.encode(prompt)
        if not tokens:
            continue

        input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(1).to(device)
        times = []
        generated_lengths = []

        for _ in range(num_runs):
            with torch.no_grad():
                current_ids = input_ids.clone()
                start_time = time.time()

                for _ in range(max_len):
                    attn_mask = generate_square_subsequent_mask(current_ids.size(0), device)
                    logits = model(current_ids, attn_mask)
                    next_token_logits = logits[-1, 0, :] / 0.8
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1)

                    if next_token.item() == tokenizer.eof_id:
                        break

                    current_ids = torch.cat([current_ids, next_token.unsqueeze(0)], dim=0)

                end_time = time.time()
                times.append(end_time - start_time)
                generated_lengths.append(current_ids.size(0) - len(tokens))

        avg_time = sum(times) / len(times)
        avg_length = sum(generated_lengths) / len(generated_lengths)
        results.append({
            'prompt': prompt,
            'avg_time': avg_time,
            'avg_generated_tokens': avg_length,
            'tokens_per_sec': avg_length / avg_time if avg_time > 0 else 0
        })

    return results


def compare_all_models(original_model, quantized_model, pruned_model, tokenizer, test_prompts, device='cpu'):
    print("\n" + "="*60)
    print("性能对比测试")
    print("="*60)

    models = {
        '原始模型': original_model,
        '量化模型': quantized_model,
        '剪枝模型': pruned_model
    }

    all_results = {}
    for name, model in models.items():
        print(f"\n测试 {name}...")
        results = benchmark_inference(model, tokenizer, test_prompts, device)
        all_results[name] = results

    print("\n" + "="*60)
    print("推理性能对比 (前向传播)")
    print("="*60)
    print(f"{'模型':<15} {'平均耗时(ms)':<15} {'吞吐量(tokens/s)':<15}")
    print("-"*45)

    for name, results in all_results.items():
        avg_time = sum(r['avg_time'] for r in results) / len(results) * 1000
        avg_throughput = sum(r['tokens_per_sec'] for r in results) / len(results)
        print(f"{name:<15} {avg_time:<15.2f} {avg_throughput:<15.2f}")

    return all_results


def test_optimization_levels(model_path, vocab_path, device='cpu'):
    print("\n" + "="*60)
    print("不同优化级别对比")
    print("="*60)

    original_model, model_args = load_model(model_path, device)
    tokenizer = CharTokenizer.load(vocab_path)

    test_prompts = ["你好", "什么是人工智能", "今天天气怎么样"]

    original_size = get_model_size(original_model)
    print(f"\n原始模型大小: {original_size:.2f} MB")

    optimized_models = {}

    print("\n[1/3] 轻量级优化 (仅量化)...")
    light_model = dynamic_quantization(original_model)
    light_size = get_model_size(light_model)
    print(f"轻量级优化后: {light_size:.2f} MB (压缩 {original_size/light_size:.2f}x)")
    optimized_models['轻量级'] = light_model

    print("\n[2/3] 中级优化 (量化 + 剪枝)...")
    medium_model, _ = load_model(model_path, device)
    medium_model = structured_pruning(medium_model, amount=0.3, method='l1')
    medium_model = structured_pruning_heads(medium_model, heads_to_prune=2)
    medium_model = remove_pruning_reparametrization(medium_model)
    medium_model = dynamic_quantization(medium_model)
    medium_size = get_model_size(medium_model)
    print(f"中级优化后: {medium_size:.2f} MB (压缩 {original_size/medium_size:.2f}x)")
    optimized_models['中级'] = medium_model

    print("\n[3/3] 激进优化 (量化 + 剪枝)...")
    aggressive_model, _ = load_model(model_path, device)
    aggressive_model = structured_pruning(aggressive_model, amount=0.5, method='l1')
    aggressive_model = structured_pruning_heads(aggressive_model, heads_to_prune=3)
    aggressive_model = remove_pruning_reparametrization(aggressive_model)
    aggressive_model = dynamic_quantization(aggressive_model)
    aggressive_size = get_model_size(aggressive_model)
    print(f"激进优化后: {aggressive_size:.2f} MB (压缩 {original_size/aggressive_size:.2f}x)")
    optimized_models['激进'] = aggressive_model

    print("\n" + "="*60)
    print("优化级别对比汇总")
    print("="*60)
    print(f"{'优化级别':<15} {'模型大小(MB)':<15} {'压缩比':<15}")
    print("-"*45)
    print(f"{'原始':<15} {original_size:<15.2f} {'1.00x':<15}")
    print(f"{'轻量级':<15} {light_size:<15.2f} {original_size/light_size:<15.2f}x")
    print(f"{'中级':<15} {medium_size:<15.2f} {original_size/medium_size:<15.2f}x")
    print(f"{'激进':<15} {aggressive_size:<15.2f} {original_size/aggressive_size:<15.2f}x")

    return optimized_models


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    tokenizer = CharTokenizer.load(args.vocab_path)

    if args.compare_levels:
        test_optimization_levels(args.model_path, args.vocab_path, device)
        return

    original_model, model_args = load_model(args.model_path, device)

    print("\n创建优化模型...")
    quantized_model = dynamic_quantization(original_model)

    pruned_model, _ = load_model(args.model_path, device)
    pruned_model = structured_pruning(pruned_model, amount=0.3, method='l1')
    pruned_model = structured_pruning_heads(pruned_model, heads_to_prune=2)
    pruned_model = remove_pruning_reparametrization(pruned_model)

    test_prompts = ["你好", "什么是人工智能", "今天天气怎么样", "解释一下深度学习"]

    print("\n" + "="*60)
    print("模型大小对比")
    print("="*60)
    print(f"原始模型: {get_model_size(original_model):.2f} MB")
    print(f"量化模型: {get_model_size(quantized_model):.2f} MB")
    print(f"剪枝模型: {get_model_size(pruned_model):.2f} MB")

    compare_all_models(original_model, quantized_model, pruned_model, tokenizer, test_prompts, device)

    print("\n" + "="*60)
    print("生成性能对比")
    print("="*60)

    for name, model in [('原始', original_model), ('量化', quantized_model), ('剪枝', pruned_model)]:
        print(f"\n测试 {name}模型生成性能...")
        results = benchmark_generation(model, tokenizer, test_prompts, device, max_len=50)
        avg_tokens_per_sec = sum(r['tokens_per_sec'] for r in results) / len(results)
        print(f"  平均生成速度: {avg_tokens_per_sec:.2f} tokens/s")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='性能测试')
    parser.add_argument('--model_path', type=str, required=True, help='模型路径')
    parser.add_argument('--vocab_path', type=str, required=True, help='词表路径')
    parser.add_argument('--compare_levels', action='store_true', help='对比不同优化级别')
    args = parser.parse_args()

    main(args)
