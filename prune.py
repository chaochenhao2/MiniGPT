"""
模型剪枝模块 - 支持结构化和非结构化剪枝
用法：python prune.py --model_path ./my_model.pt --vocab_path ./vocab.json --output_path ./model_pruned.pt
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from train_llm import MiniGPT, CharTokenizer, generate_square_subsequent_mask


def load_model(model_path, device='cpu'):
    checkpoint = torch.load(model_path, map_location=device)
    model_args = checkpoint['model_args']
    model = MiniGPT(**model_args)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, model_args


def get_model_size(model):
    model_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    return model_size / 1024 / 1024


def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    pruned = sum(p.numel() - p.count_nonzero().item() for p in model.parameters())
    return total, pruned, pruned / total if total > 0 else 0


def analyze_importance(model):
    importance_info = []
    for name, param in model.named_parameters():
        if 'weight' in name and param.dim() >= 2:
            importance = param.data.abs().mean().item()
            sparsity = (param.data == 0).float().mean().item()
            importance_info.append({
                'name': name,
                'importance': importance,
                'sparsity': sparsity,
                'shape': list(param.shape)
            })
    return importance_info


def structured_pruning(model, amount=0.3, method='l1'):
    for name, module in model.named_modules():
        if isinstance(module, nn.MultiheadAttention):
            if hasattr(module, 'in_proj_weight') and module.in_proj_weight is not None:
                if method == 'l1':
                    prune.l1_unstructured(module, name='in_proj_weight', amount=amount)
                elif method == 'random':
                    prune.random_unstructured(module, name='in_proj_weight', amount=amount)

            if hasattr(module, 'out_proj') and module.out_proj is not None:
                if method == 'l1':
                    prune.l1_unstructured(module.out_proj, name='weight', amount=amount)
                elif method == 'random':
                    prune.random_unstructured(module.out_proj, name='weight', amount=amount)

        elif isinstance(module, nn.Linear):
            if method == 'l1':
                prune.l1_unstructured(module, name='weight', amount=amount)
            elif method == 'random':
                prune.random_unstructured(module, name='weight', amount=amount)

    return model


def structured_pruning_heads(model, heads_to_prune=2):
    for layer_idx, layer in enumerate(model.layers):
        attn = layer.self_attn
        if hasattr(attn, 'in_proj_weight') and attn.in_proj_weight is not None:
            d_model = attn.in_proj_weight.shape[0] // 3
            nhead = 8
            head_dim = d_model // nhead

            qkv_weight = attn.in_proj_weight
            q_weight = qkv_weight[:d_model]
            k_weight = qkv_weight[d_model:2*d_model]
            v_weight = qkv_weight[2*d_model:]

            q_reshaped = q_weight.view(nhead, head_dim, d_model)
            k_reshaped = k_weight.view(nhead, head_dim, d_model)
            v_reshaped = v_weight.view(nhead, head_dim, d_model)

            head_importance = (
                q_reshaped.abs().sum(dim=[1, 2]) +
                k_reshaped.abs().sum(dim=[1, 2]) +
                v_reshaped.abs().sum(dim=[1, 2])
            )

            least_important = head_importance.argsort()[:heads_to_prune]

            for head_idx in least_important:
                q_reshaped[head_idx] = 0
                k_reshaped[head_idx] = 0
                v_reshaped[head_idx] = 0

            attn.in_proj_weight.data = torch.cat([
                q_reshaped.view(d_model, d_model),
                k_reshaped.view(d_model, d_model),
                v_reshaped.view(d_model, d_model)
            ], dim=0)

    return model


def unstructured_global_pruning(model, amount=0.3, method='l1'):
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            parameters_to_prune.append((module, 'weight'))
        elif isinstance(module, nn.MultiheadAttention):
            if hasattr(module, 'in_proj_weight') and module.in_proj_weight is not None:
                parameters_to_prune.append((module, 'in_proj_weight'))
            if hasattr(module, 'out_proj') and module.out_proj is not None:
                parameters_to_prune.append((module.out_proj, 'weight'))

    if method == 'l1':
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.L1Unstructured,
            amount=amount,
        )
    elif method == 'random':
        prune.global_unstructured(
            parameters_to_prune,
            pruning_method=prune.RandomUnstructured,
            amount=amount,
        )

    return model


def remove_pruning_reparametrization(model):
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            if hasattr(module, 'weight_orig'):
                prune.remove(module, 'weight')
        elif isinstance(module, nn.MultiheadAttention):
            if hasattr(module, 'in_proj_weight') and hasattr(module, 'in_proj_weight_orig'):
                prune.remove(module, 'in_proj_weight')
            if hasattr(module, 'out_proj') and hasattr(module, 'out_proj_weight_orig'):
                prune.remove(module.out_proj, 'weight')
    return model


def compare_pruned_model(original_model, pruned_model, tokenizer, test_prompts, device='cpu'):
    original_model.eval()
    pruned_model.eval()

    results = []
    for prompt in test_prompts:
        tokens = tokenizer.encode(prompt)
        if not tokens:
            continue

        input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(1).to(device)

        with torch.no_grad():
            attn_mask = generate_square_subsequent_mask(input_ids.size(0), device)

            start_time = time.time()
            original_logits = original_model(input_ids, attn_mask)
            original_time = time.time() - start_time

            start_time = time.time()
            pruned_logits = pruned_model(input_ids, attn_mask)
            pruned_time = time.time() - start_time

        logits_diff = (original_logits - pruned_logits).abs().mean().item()

        results.append({
            'prompt': prompt,
            'original_time': original_time,
            'pruned_time': pruned_time,
            'logits_diff': logits_diff,
            'speedup': original_time / pruned_time if pruned_time > 0 else 0
        })

    return results


def prune_model(model_path, vocab_path, output_path, sparsity=0.3, method='l1', structured=False):
    device = torch.device('cpu')
    model, model_args = load_model(model_path, device)
    tokenizer = CharTokenizer.load(vocab_path)

    print(f"原始模型大小: {get_model_size(model):.2f} MB")
    total, pruned_before, sparsity_before = count_parameters(model)
    print(f"原始参数: {total:,}, 非零参数: {total-pruned_before:,}, 稀疏度: {sparsity_before:.2%}")

    if structured:
        print(f"\n执行结构化剪枝 (比例: {sparsity:.0%})")
        pruned_model = structured_pruning(model, amount=sparsity, method=method)
        print("执行注意力头剪枝 (每层剪2个头)")
        pruned_model = structured_pruning_heads(pruned_model, heads_to_prune=2)
    else:
        print(f"\n执行非结构化全局剪枝 (比例: {sparsity:.0%})")
        pruned_model = unstructured_global_pruning(model, amount=sparsity, method=method)

    print(f"\n剪枝后模型大小: {get_model_size(pruned_model):.2f} MB")
    total, pruned_after, sparsity_after = count_parameters(pruned_model)
    print(f"剪枝后参数: {total:,}, 非零参数: {total-pruned_after:,}, 稀疏度: {sparsity_after:.2%}")

    print("\n=== 各层稀疏度分析 ===")
    for name, param in pruned_model.named_parameters():
        if 'weight' in name:
            layer_sparsity = (param.data == 0).float().mean().item()
            if layer_sparsity > 0:
                print(f"  {name}: {layer_sparsity:.2%}")

    test_prompts = ["你好", "什么是人工智能", "今天天气怎么样"]
    results = compare_pruned_model(model, pruned_model, tokenizer, test_prompts, device)

    print("\n=== 剪枝效果对比 ===")
    for r in results:
        print(f"提示: {r['prompt']}")
        print(f"  原始耗时: {r['original_time']*1000:.2f}ms")
        print(f"  剪枝耗时: {r['pruned_time']*1000:.2f}ms")
        print(f"  加速比: {r['speedup']:.2f}x")
        print(f"  Logits差异: {r['logits_diff']:.6f}")
        print()

    pruned_model = remove_pruning_reparametrization(pruned_model)

    checkpoint = {
        'model_state_dict': pruned_model.state_dict(),
        'model_args': model_args,
        'pruned': True,
        'pruning_method': method,
        'sparsity': sparsity_after
    }
    torch.save(checkpoint, output_path)
    print(f"剪枝模型已保存至: {output_path}")

    return pruned_model, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='模型剪枝')
    parser.add_argument('--model_path', type=str, required=True, help='原始模型路径')
    parser.add_argument('--vocab_path', type=str, required=True, help='词表路径')
    parser.add_argument('--output_path', type=str, default='./model_pruned.pt', help='剪枝模型保存路径')
    parser.add_argument('--sparsity', type=float, default=0.3, help='剪枝比例 (0.0-1.0)')
    parser.add_argument('--method', type=str, default='l1', choices=['l1', 'random'], help='剪枝方法')
    parser.add_argument('--structured', action='store_true', help='使用结构化剪枝')
    args = parser.parse_args()

    prune_model(args.model_path, args.vocab_path, args.output_path, args.sparsity, args.method, args.structured)
