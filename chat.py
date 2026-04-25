"""
加载训练好的模型进行交互式对话（支持流式输出）。
用法：python chat.py --model_path ./model.pt --vocab_path ./vocab.json
"""

import argparse
import json
import math
import torch
import torch.nn.functional as F
from train_llm import MiniGPT, CharTokenizer, generate_square_subsequent_mask

def top_k_top_p_filtering(logits, top_k=0, top_p=0.0, filter_value=-float('Inf')):
    """对 logits 进行 top-k 和 top-p 过滤"""
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
    """
    流式生成器：每产生一个 token 就 yield 其对应的字符。
    当生成 <EOF> 或达到 max_len 时停止。
    """
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

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载分词器
    tokenizer = CharTokenizer.load(args.vocab_path)
    print(f"词表大小: {tokenizer.vocab_size}")

    # 加载模型
    checkpoint = torch.load(args.model_path, map_location=device)
    model_args = checkpoint['model_args']
    model = MiniGPT(**model_args).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])

    # 扩展位置编码以支持更长的生成序列
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

    model.eval()
    print("模型加载成功！")

    print("\n开始对话（输入 'quit' 退出）")
    while True:
        prompt = input("\n你: ")
        if prompt.lower() == 'quit':
            break

        # 防止用户输入尖括号破坏解析
        safe_prompt = prompt.replace('<', '＜').replace('>', '＞')

        print("AI: ", end='', flush=True)

        for chunk in generate_stream(
            model, tokenizer, safe_prompt,
            max_len=args.max_len,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=device
        ):
            print(chunk, end='', flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='与训练好的语言模型对话（流式输出）')
    parser.add_argument('--model_path', type=str, required=True, help='模型文件路径（.pt）')
    parser.add_argument('--vocab_path', type=str, required=True, help='分词器文件路径（.json）')
    parser.add_argument('--max_len', type=int, default=200, help='最大生成长度')
    parser.add_argument('--temperature', type=float, default=0.8, help='温度系数')
    parser.add_argument('--top_k', type=int, default=40, help='Top-k 采样参数')
    parser.add_argument('--top_p', type=float, default=0.9, help='Top-p (nucleus) 采样参数')
    args = parser.parse_args()
    main(args)