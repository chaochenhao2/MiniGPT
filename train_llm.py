"""
从零训练一个基于 Transformer 解码器的语言模型（支持 <EOF> 和 <UNK>）。
用法：python train_llm.py --data_path ./data.txt --save_model ./my_model.pt --save_vocab ./vocab.json
"""

import argparse
import json
import math
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# -------------------- 模型定义 --------------------
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0)].unsqueeze(1)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout1(attn_out)
        x = self.norm1(x)
        ff_out = self.linear2(self.dropout(F.relu(self.linear1(x))))
        x = x + self.dropout2(ff_out)
        x = self.norm2(x)
        return x

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)
        ])
        self.ln_final = nn.LayerNorm(d_model)
        self.fc_out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)
        self._init_weights()

    def _init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x, attn_mask=None):
        x = self.embedding(x) * math.sqrt(self.d_model)
        x = self.pos_encoder(x)
        x = self.dropout(x)
        for layer in self.layers:
            x = layer(x, attn_mask)
        x = self.ln_final(x)
        return self.fc_out(x)

def generate_square_subsequent_mask(sz, device):
    mask = torch.triu(torch.ones(sz, sz, device=device) * float('-inf'), diagonal=1)
    return mask

# -------------------- 数据集与分词 --------------------
class CharTokenizer:
    """字符级分词器，支持 <EOF> 结束标记和 <UNK> 未知字符标记"""
    def __init__(self):
        self.char2idx = {}
        self.idx2char = {}
        self.vocab_size = 0
        self.eof_token = '<EOF>'
        self.unk_token = '<UNK>'
        self.eof_id = None
        self.unk_id = None

    def fit(self, text):
        # 移除特殊标记以统计基础字符
        temp_text = text.replace(self.eof_token, '').replace(self.unk_token, '')
        chars = sorted(list(set(temp_text)))
        # 加入特殊标记
        for token in [self.eof_token, self.unk_token]:
            if token not in chars:
                chars.append(token)
        self.char2idx = {ch: i for i, ch in enumerate(chars)}
        self.idx2char = {i: ch for ch, i in self.char2idx.items()}
        self.vocab_size = len(chars)
        self.eof_id = self.char2idx[self.eof_token]
        self.unk_id = self.char2idx[self.unk_token]

    def encode(self, text):
        ids = []
        i = 0
        eof_len = len(self.eof_token)
        unk_len = len(self.unk_token)
        while i < len(text):
            if text[i:i+eof_len] == self.eof_token:
                ids.append(self.eof_id)
                i += eof_len
            elif text[i:i+unk_len] == self.unk_token:
                ids.append(self.unk_id)
                i += unk_len
            else:
                ch = text[i]
                if ch in self.char2idx:
                    ids.append(self.char2idx[ch])
                else:
                    ids.append(self.unk_id)   # 未知字符映射到 <UNK>
                i += 1
        return ids

    def decode(self, indices):
        return ''.join([self.idx2char.get(idx, self.unk_token) for idx in indices])

    def save(self, path):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump({
                'char2idx': self.char2idx,
                'idx2char': {str(k): v for k, v in self.idx2char.items()},
                'eof_token': self.eof_token,
                'unk_token': self.unk_token
            }, f, ensure_ascii=False)

    @classmethod
    def load(cls, path):
        tokenizer = cls()
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        tokenizer.char2idx = data['char2idx']
        tokenizer.idx2char = {int(k): v for k, v in data['idx2char'].items()}
        tokenizer.vocab_size = len(tokenizer.char2idx)
        tokenizer.eof_token = data.get('eof_token', '<EOF>')
        tokenizer.unk_token = data.get('unk_token', '<UNK>')
        tokenizer.eof_id = tokenizer.char2idx[tokenizer.eof_token]
        tokenizer.unk_id = tokenizer.char2idx[tokenizer.unk_token]
        return tokenizer

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, seq_len=128):
        self.seq_len = seq_len
        self.tokens = tokenizer.encode(text)
        self.num_samples = max(len(self.tokens) - seq_len, 0)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        chunk = self.tokens[idx:idx + self.seq_len + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y

# -------------------- 训练主函数 --------------------
def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    with open(args.data_path, 'r', encoding='utf-8') as f:
        text = f.read()
    print(f"文本总长度: {len(text)} 字符")

    # 构建或加载分词器
    if args.vocab_path and os.path.exists(args.vocab_path):
        tokenizer = CharTokenizer.load(args.vocab_path)
        print(f"从 {args.vocab_path} 加载分词器，词表大小: {tokenizer.vocab_size}")
    else:
        tokenizer = CharTokenizer()
        tokenizer.fit(text)
        if args.save_vocab:
            tokenizer.save(args.save_vocab)
            print(f"分词器已保存至 {args.save_vocab}")
        print(f"新建分词器，词表大小: {tokenizer.vocab_size}，EOF ID: {tokenizer.eof_id}，UNK ID: {tokenizer.unk_id}")

    dataset = TextDataset(text, tokenizer, seq_len=args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"数据集样本数: {len(dataset)}")

    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        nhead=args.nhead,
        num_layers=args.num_layers,
        dim_feedforward=args.dim_ff,
        max_len=args.seq_len,
        dropout=args.dropout
    ).to(device)
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f} M")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for x, y in pbar:
            x, y = x.to(device).transpose(0, 1), y.to(device).transpose(0, 1)
            attn_mask = generate_square_subsequent_mask(x.size(0), device)

            optimizer.zero_grad()
            logits = model(x, attn_mask)
            loss = criterion(logits.reshape(-1, tokenizer.vocab_size), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} 平均损失: {avg_loss:.4f}")

        if args.save_model:
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'model_args': {
                    'vocab_size': tokenizer.vocab_size,
                    'd_model': args.d_model,
                    'nhead': args.nhead,
                    'num_layers': args.num_layers,
                    'dim_feedforward': args.dim_ff,
                    'max_len': args.seq_len,
                    'dropout': args.dropout
                }
            }
            torch.save(checkpoint, args.save_model)
            print(f"模型已保存至 {args.save_model}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='从零训练语言模型')
    parser.add_argument('--data_path', type=str, required=True, help='训练文本文件路径')
    parser.add_argument('--save_model', type=str, default='./model.pt', help='模型保存路径')
    parser.add_argument('--save_vocab', type=str, default='./vocab.json', help='分词器保存路径')
    parser.add_argument('--vocab_path', type=str, default=None, help='已有分词器路径（可选）')
    parser.add_argument('--seq_len', type=int, default=64, help='训练序列长度')
    parser.add_argument('--batch_size', type=int, default=16, help='批次大小')
    parser.add_argument('--epochs', type=int, default=20, help='训练轮数')
    parser.add_argument('--lr', type=float, default=3e-4, help='学习率')
    parser.add_argument('--d_model', type=int, default=256, help='嵌入维度')
    parser.add_argument('--nhead', type=int, default=8, help='注意力头数')
    parser.add_argument('--num_layers', type=int, default=4, help='Transformer 层数')
    parser.add_argument('--dim_ff', type=int, default=1024, help='前馈网络维度')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout 比率')
    parser.add_argument('--grad_clip', type=float, default=1.0, help='梯度裁剪阈值')
    args = parser.parse_args()
    train(args)