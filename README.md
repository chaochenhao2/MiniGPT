# MiniGPT

从零实现的 Transformer 解码器语言模型，包含训练、推理和边缘设备优化脚本。

## 项目结构

```
.
├── src/                # 源码目录
│   ├── train_llm.py    # 模型训练脚本
│   ├── chat.py         # 交互式对话脚本（支持流式输出和KV Cache优化）
│   ├── quantize.py     # 模型量化模块
│   ├── prune.py        # 模型剪枝模块
│   ├── optimize.py     # 模型优化整合脚本
│   └── benchmark.py    # 性能测试脚本
├── scripts/            # 脚本目录
│   ├── train.sh        # 训练启动脚本
│   ├── chat.sh         # 对话启动脚本
│   ├── add.sh          # 追加训练脚本
│   ├── update.sh       # 重新训练脚本
│   └── optimize.sh     # 优化启动脚本
├── data/               # 数据目录
│   ├── data.txt        # 训练语料
│   ├── data_new.txt    # 追加训练语料
│   └── vocab.json      # 词表文件
├── models/             # 模型目录
│   └── my_model.pt     # 训练好的模型
├── README.md
├── LICENSE
└── .gitignore
```

## 特性

- **分词器**: 字符级分词器，支持 `<EOF>` 结束标记和 `<UNK>` 未知字符标记
- **模型架构**: Transformer 解码器，支持自定义层数、注意力头数、嵌入维度
- **位置编码**: Sinusoidal 位置编码，支持扩展到更长序列
- **采样策略**: Top-k + Top-p (Nucleus) 采样，支持温度调节
- **训练特性**: 梯度裁剪、AdamW 优化器、可配置 Dropout
- **追加训练**: 支持在已有模型基础上继续训练
- **边缘优化**: 支持模型量化、剪枝和推理优化，适配边缘设备部署

## 快速开始

### 1. 安装依赖
```bash
pip install torch tqdm
```

### 2. 准备数据
编辑 `data/data.txt`，放入你的训练语料（每段结尾加 `<EOF>`）

### 3. 训练模型

```bash
# 使用默认参数训练
bash scripts/train.sh

# 或手动指定参数
bash scripts/train.sh 5 16 64 0.0005
```

训练参数：
- 参数1: epochs（默认 3）
- 参数2: batch_size（默认 8）
- 参数3: seq_len（默认 32）
- 参数4: lr（默认 0.001）

### 4. 对话测试

```bash
# 使用启动脚本
bash scripts/chat.sh

# 或手动指定
python3 src/chat.py --model_path ./models/my_model.pt --vocab_path ./data/vocab.json
```

可选参数：
- `--max_len`: 最大生成长度（默认 200）
- `--temperature`: 温度系数，越高越随机（默认 0.8）
- `--top_k`: Top-k 采样（默认 40）
- `--top_p`: Top-p (Nucleus) 采样（默认 0.9）
- `--use_cache`: 启用 KV Cache 优化（推荐）
- `--use_compile`: 启用 torch.compile 优化

### 5. 追加训练

当有新数据需要加入训练时，使用 `add.sh` 脚本：

```bash
bash scripts/add.sh
```

该脚本会：
1. 使用 `data/data_new.txt` 中的数据对模型进行追加训练
2. 将新数据追加到 `data/data.txt` 并清空 `data/data_new.txt`
3. 启动对话模式

### 重新训练

如果需要完全重新训练（删除已有模型和词表）：

```bash
bash scripts/update.sh
```

## 边缘设备优化

本项目支持多种优化方案，适用于边缘设备部署：

### 优化级别

| 级别 | 说明 | 模型大小 | 压缩比 |
|------|------|----------|--------|
| light | 仅量化 | ~3.37 MB | 4x |
| medium | 量化 + 剪枝 | ~2-3 MB | 5-6x |
| aggressive | 量化 + 剪枝 | ~1.5-2 MB | 7-9x |

### 快速优化

```bash
# 轻量级优化 (仅量化)
bash scripts/optimize.sh light

# 中级优化 (量化 + 剪枝)
bash scripts/optimize.sh medium

# 激进优化 (量化 + 剪枝)
bash scripts/optimize.sh aggressive

# 性能测试
bash scripts/optimize.sh benchmark

# 使用优化模型对话
bash scripts/optimize.sh chat
```

### 单独使用优化模块

```bash
# 模型量化
python3 src/quantize.py --model_path ./models/my_model.pt --vocab_path ./data/vocab.json --output_path ./models/model_quantized.pt

# 模型剪枝
python3 src/prune.py --model_path ./models/my_model.pt --vocab_path ./data/vocab.json --output_path ./models/model_pruned.pt --sparsity 0.3

# 性能测试
python3 src/benchmark.py --model_path ./models/my_model.pt --vocab_path ./data/vocab.json --compare_levels
```

### 优化说明

1. **模型量化 (quantize.py)**
   - 支持动态量化和静态量化
   - 将 FP32 权重转换为 INT8
   - 压缩模型大小 4 倍
   - 推理速度提升 2-4 倍

2. **模型剪枝 (prune.py)**
   - 支持结构化和非结构化剪枝
   - 移除冗余注意力头和神经元
   - 减少参数量 30%-50%
   - 推理速度提升 1.5-2 倍

3. **推理优化 (chat.py)**
   - KV Cache: 避免重复计算历史键值对
   - torch.compile: 图优化和算子融合
   - 推理延迟降低 30%-50%

### 性能测试

运行性能测试对比不同优化方案：

```bash
python3 src/benchmark.py --model_path ./models/my_model.pt --vocab_path ./data/vocab.json --compare_levels
```

测试内容包括：
- 模型大小对比
- 推理速度对比
- 生成速度对比
- 不同优化级别对比

## 环境依赖

- Python 3.8+
- PyTorch 2.0+
- tqdm
