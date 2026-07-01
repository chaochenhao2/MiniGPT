"""
模型量化模块 - 支持动态量化和静态量化
用法：python quantize.py --model_path ./my_model.pt --vocab_path ./vocab.json --output_path ./model_quantized.pt
"""

import argparse
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from train_llm import MiniGPT, CharTokenizer, TextDataset, generate_square_subsequent_mask


def load_model(model_path, device='cpu'):
    checkpoint = torch.load(model_path, map_location=device)
    model_args = checkpoint['model_args']
    model = MiniGPT(**model_args)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model, model_args


def get_model_size(model):
    model_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    return model_size / 1024 / 1024


def dynamic_quantization(model):
    model_cpu = model.cpu()
    quantized_model = torch.quantization.quantize_dynamic(
        model_cpu,
        {nn.Linear, nn.Embedding},
        dtype=torch.qint8
    )
    return quantized_model


def static_quantization(model, calibration_data=None, tokenizer=None, seq_len=32):
    model_cpu = model.cpu()
    model_cpu.eval()

    model_cpu.qconfig = torch.quantization.get_default_qconfig('fbgemm')
    torch.quantization.prepare(model_cpu, inplace=True)

    if calibration_data is None and tokenizer is not None:
        dataset = TextDataset(calibration_data, tokenizer, seq_len=seq_len)
        dataloader = DataLoader(dataset, batch_size=1, shuffle=False)
        with torch.no_grad():
            for x, y in dataloader:
                x = x.transpose(0, 1)
                attn_mask = generate_square_subsequent_mask(x.size(0), x.device)
                model_cpu(x, attn_mask)

    torch.quantization.convert(model_cpu, inplace=True)
    return model_cpu


def compare_models(original_model, quantized_model, tokenizer, test_prompts, device='cpu'):
    original_model.eval()
    quantized_model.eval()

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
            quantized_logits = quantized_model(input_ids, attn_mask)
            quantized_time = time.time() - start_time

        logits_diff = (original_logits - quantized_logits).abs().mean().item()

        results.append({
            'prompt': prompt,
            'original_time': original_time,
            'quantized_time': quantized_time,
            'logits_diff': logits_diff,
            'speedup': original_time / quantized_time if quantized_time > 0 else 0
        })

    return results


def quantize_model(model_path, vocab_path, output_path, method='dynamic'):
    device = torch.device('cpu')
    model, model_args = load_model(model_path, device)
    tokenizer = CharTokenizer.load(vocab_path)

    print(f"原始模型大小: {get_model_size(model):.2f} MB")

    if method == 'dynamic':
        quantized_model = dynamic_quantization(model)
    else:
        quantized_model = static_quantization(model, tokenizer=tokenizer)

    print(f"量化后模型大小: {get_model_size(quantized_model):.2f} MB")

    test_prompts = ["你好", "什么是人工智能", "今天天气怎么样"]
    results = compare_models(model, quantized_model, tokenizer, test_prompts, device)

    print("\n=== 量化效果对比 ===")
    for r in results:
        print(f"提示: {r['prompt']}")
        print(f"  原始耗时: {r['original_time']*1000:.2f}ms")
        print(f"  量化耗时: {r['quantized_time']*1000:.2f}ms")
        print(f"  加速比: {r['speedup']:.2f}x")
        print(f"  Logits差异: {r['logits_diff']:.6f}")
        print()

    checkpoint = {
        'model_state_dict': quantized_model.state_dict(),
        'model_args': model_args,
        'quantized': True,
        'quantization_method': method
    }
    torch.save(checkpoint, output_path)
    print(f"量化模型已保存至: {output_path}")

    return quantized_model, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='模型量化')
    parser.add_argument('--model_path', type=str, required=True, help='原始模型路径')
    parser.add_argument('--vocab_path', type=str, required=True, help='词表路径')
    parser.add_argument('--output_path', type=str, default='./model_quantized.pt', help='量化模型保存路径')
    parser.add_argument('--method', type=str, default='dynamic', choices=['dynamic', 'static'], help='量化方法')
    args = parser.parse_args()

    quantize_model(args.model_path, args.vocab_path, args.output_path, args.method)
