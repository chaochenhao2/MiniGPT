"""
加载训练好的模型进行交互式对话（支持流式输出和KV Cache优化）
用法：python chat.py --model_path ./model.pt --vocab_path ./vocab.json
"""

import argparse
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from train_llm import MiniGPT, CharTokenizer, generate_square_subsequent_mask
from quantize import dynamic_quantization


def manual_attention(x, k, v, attn_mask=None, nhead=8):
    seq_len_q, batch_size, d_model = x.shape
    seq_len_k = k.size(0)
    head_dim = d_model // nhead

    q = x.view(seq_len_q, batch_size, nhead, head_dim).permute(1, 2, 0, 3)
    k_full = k.view(seq_len_k, batch_size, nhead, head_dim).permute(1, 2, 0, 3)
    v_full = v.view(seq_len_k, batch_size, nhead, head_dim).permute(1, 2, 0, 3)

    scale = math.sqrt(head_dim)
    scores = torch.matmul(q, k_full.transpose(-2, -1)) / scale

    if attn_mask is not None:
        scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)

    attn_weights = F.softmax(scores, dim=-1)
    attn_out = torch.matmul(attn_weights, v_full)

    attn_out = attn_out.permute(2, 0, 1, 3).contiguous().view(seq_len_q, batch_size, d_model)
    return attn_out


class MiniGPTWithKVCache(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.original_model = original_model
        self.kv_cache = {}
        self.nhead = original_model.layers[0].self_attn.num_heads

    def forward(self, x, attn_mask=None, use_cache=False):
        if not use_cache:
            return self.original_model(x, attn_mask)

        embedding = self.original_model.embedding(x) * math.sqrt(self.original_model.d_model)
        pos_encoding = self.original_model.pos_encoder(embedding)
        x = self.original_model.dropout(pos_encoding)

        for i, layer in enumerate(self.original_model.layers):
            layer_key = f'layer_{i}'
            has_cache = layer_key in self.kv_cache

            if hasattr(layer.self_attn, 'in_proj_weight') and layer.self_attn.in_proj_weight is not None:
                qkv_weight = layer.self_attn.in_proj_weight
                d_model = x.size(-1)
                q_weight = qkv_weight[:d_model]
                k_weight = qkv_weight[d_model:2*d_model]
                v_weight = qkv_weight[2*d_model:]

                q = x @ q_weight.T
                k_new = x @ k_weight.T
                v_new = x @ v_weight.T
            else:
                q = x
                k_new = x
                v_new = x

            if has_cache:
                k_cache, v_cache = self.kv_cache[layer_key]
                k = torch.cat([k_cache, k_new], dim=0)
                v = torch.cat([v_cache, v_new], dim=0)
            else:
                k = k_new
                v = v_new

            if use_cache:
                self.kv_cache[layer_key] = (k, v)

            attn_out = manual_attention(x, k, v, attn_mask, self.nhead)
            attn_out = layer.self_attn.out_proj(attn_out)

            x = x + layer.dropout1(attn_out)
            x = layer.norm1(x)
            ff_out = layer.linear2(layer.dropout(F.relu(layer.linear1(x))))
            x = x + layer.dropout2(ff_out)
            x = layer.norm2(x)

        x = self.original_model.ln_final(x)
        return self.original_model.fc_out(x)

    def clear_cache(self):
        self.kv_cache = {}


def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    assert logits.dim() == 1
    top_k = min(top_k, logits.size(-1))
    if top_k > 0:
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = filter_value

    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices[sorted_indices_to_remove]
        logits[indices_to_remove] = filter_value
    return logits


def generate_stream(model, tokenizer, prompt, max_len=100, temperature=1.0, top_k=0, top_p=0.9, device='cpu'):
    model.eval()
    eof_id = tokenizer.eof_id
    tokens = tokenizer.encode(prompt)
    if len(tokens) == 0:
        yield " [无法识别]\n"
        return

    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(1).to(device)

    with torch.no_grad():
        for _ in range(max_len):
            attn_mask = generate_square_subsequent_mask(input_ids.size(0), device)
            logits = model(input_ids, attn_mask)
            next_token_logits = logits[-1, 0, :] / temperature

            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if next_token.item() == eof_id:
                break

            input_ids = torch.cat([input_ids, next_token.unsqueeze(0)], dim=0)

            token_str = tokenizer.decode([next_token.item()])
            clean_str = token_str.replace('<UNK>', '?')
            yield clean_str

    yield "\n"


def generate_stream_optimized(model, tokenizer, prompt, max_len=100, temperature=1.0, top_k=0, top_p=0.9, device='cpu'):
    model.eval()
    eof_id = tokenizer.eof_id
    tokens = tokenizer.encode(prompt)
    if len(tokens) == 0:
        yield " [无法识别]\n"
        return

    input_ids = torch.tensor(tokens, dtype=torch.long).unsqueeze(1).to(device)

    with torch.no_grad():
        logits = model(input_ids, generate_square_subsequent_mask(input_ids.size(0), device), use_cache=True)

        next_token_logits = logits[-1, 0, :] / temperature
        filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
        probs = F.softmax(filtered_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)

        if next_token.item() == eof_id:
            yield "\n"
            return

        token_str = tokenizer.decode([next_token.item()])
        clean_str = token_str.replace('<UNK>', '?')
        yield clean_str

        current_token = next_token.unsqueeze(1)

        for _ in range(max_len - 1):
            attn_mask = generate_square_subsequent_mask(1, device)
            logits = model(current_token, attn_mask, use_cache=True)

            next_token_logits = logits[-1, 0, :] / temperature
            filtered_logits = top_k_top_p_filtering(next_token_logits, top_k=top_k, top_p=top_p)
            probs = F.softmax(filtered_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if next_token.item() == eof_id:
                break

            token_str = tokenizer.decode([next_token.item()])
            clean_str = token_str.replace('<UNK>', '?')
            yield clean_str

            current_token = next_token.unsqueeze(1)

    yield "\n"


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    tokenizer = CharTokenizer.load(args.vocab_path)
    print(f"词表大小: {tokenizer.vocab_size}")

    checkpoint = torch.load(args.model_path, map_location=device)
    model_args = checkpoint['model_args']

    if checkpoint.get('quantized', False):
        print("检测到量化模型，重新应用量化...")
        model = MiniGPT(**model_args).to(device)
        if 'model_state_dict_original' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict_original'])
        else:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        model = dynamic_quantization(model)
        print("量化模型加载成功")
    else:
        model = MiniGPT(**model_args).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])

    old_pe = model.pos_encoder.pe
    old_max_len, d_model = old_pe.shape
    new_max_len = 5000
    if old_max_len < new_max_len:
        new_pe = torch.zeros(new_max_len, d_model)
        new_pe[:old_max_len] = old_pe
        position = torch.arange(old_max_len, new_max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        new_pe[old_max_len:, 0::2] = torch.sin(position * div_term)
        new_pe[old_max_len:, 1::2] = torch.cos(position * div_term)
        model.pos_encoder.register_buffer('pe', new_pe)
        print(f"位置编码已从 {old_max_len} 扩展到 {new_max_len}")

    if args.use_cache:
        model = MiniGPTWithKVCache(model).to(device)
        print("已启用 KV Cache 优化")

    if args.use_compile:
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print("已启用 torch.compile 优化")
        except Exception as e:
            print(f"torch.compile 不可用: {e}")

    model.eval()
    print("模型加载成功！")

    print("\n开始对话（输入 'quit' 退出）")
    while True:
        prompt = input("\n你: ")
        if prompt.lower() == 'quit':
            break

        safe_prompt = prompt.replace('<', '＜').replace('>', '＞')

        print("AI: ", end='', flush=True)

        generate_fn = generate_stream_optimized if args.use_cache else generate_stream

        for chunk in generate_fn(
            model, tokenizer, safe_prompt,
            max_len=args.max_len,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device
        ):
            print(chunk, end='', flush=True)

        if args.use_cache and hasattr(model, 'clear_cache'):
            model.clear_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='与训练好的语言模型对话（流式输出）')
    parser.add_argument('--model_path', type=str, required=True, help='模型文件路径（.pt）')
    parser.add_argument('--vocab_path', type=str, required=True, help='分词器文件路径（.json）')
    parser.add_argument('--max_len', type=int, default=200, help='最大生成长度')
    parser.add_argument('--temperature', type=float, default=0.8, help='温度系数')
    parser.add_argument('--top_k', type=int, default=40, help='Top-k 采样参数')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p (nucleus) 采样参数')
    parser.add_argument('--use_cache', action='store_true', help='启用 KV Cache 优化')
    parser.add_argument('--use_compile', action='store_true', help='启用 torch.compile 优化')
    args = parser.parse_args()
    main(args)
