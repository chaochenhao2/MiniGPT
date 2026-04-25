# MiniGPT

从零实现的 Transformer 解码器语言模型，包含训练和推理脚本。

## 项目结构

```
.
├── train_llm.py      # 模型训练脚本
├── chat.py           # 交互式对话脚本（支持流式输出）
├── vocab.json        # 词表文件
├── my_model.pt       # 训练好的模型
├── data.txt          # 训练语料
├── data_new.txt      # 追加训练语料
├── train.sh          # 训练启动脚本
├── chat.sh           # 对话启动脚本
├── add.sh            # 追加训练脚本
└── update.sh         # 重新训练脚本
```

## 特性

- **分词器**: 字符级分词器，支持 `<EOF>` 结束标记和 `<UNK>` 未知字符标记
- **模型架构**: Transformer 解码器，支持自定义层数、注意力头数、嵌入维度
- **位置编码**: Sinusoidal 位置编码，支持扩展到更长序列
- **采样策略**: Top-k + Top-p (Nucleus) 采样，支持温度调节
- **训练特性**: 梯度裁剪、AdamW 优化器、可配置 Dropout
- **追加训练**: 支持在已有模型基础上继续训练

## 快速开始

### 1. 安装依赖
``` bash
pip install torch tqdm
```

### 2. 准备数据
编辑 data.txt，放入你的训练语料（每段结尾加 `<EOF>` ）

### 3. 训练模型

```bash
# 使用默认参数训练
bash train.sh

# 或手动指定参数
python train_llm.py --data_path ./data.txt --save_model ./my_model.pt --save_vocab ./vocab.json
```

训练参数可通过以下选项调整：
- `--data_path`: 训练文本文件路径
- `--save_model`: 模型保存路径
- `--save_vocab`: 词表保存路径
- `--vocab_path`: 已有词表路径（追加训练时使用）
- `--seq_len`: 序列长度（默认 64）
- `--batch_size`: 批次大小（默认 16）
- `--epochs`: 训练轮数（默认 20）
- `--d_model`: 嵌入维度（默认 256）
- `--nhead`: 注意力头数（默认 8）
- `--num_layers`: Transformer 层数（默认 4）
- `--dim_ff`: 前馈网络维度（默认 1024）
- `--dropout`: Dropout 比率（默认 0.1）
- `--lr`: 学习率（默认 3e-4）
- `--grad_clip`: 梯度裁剪阈值（默认 1.0）

### 4. 对话测试

```bash
# 使用启动脚本
bash chat.sh

# 或手动指定
python chat.py --model_path ./my_model.pt --vocab_path ./vocab.json
```

可选参数：
- `--max_len`: 最大生成长度（默认 200）
- `--temperature`: 温度系数，越高越随机（默认 0.8）
- `--top_k`: Top-k 采样（默认 40）
- `--top_p`: Top-p (Nucleus) 采样（默认 0.9）

### 5. 追加训练

当有新数据需要加入训练时，使用 `add.sh` 脚本：

```bash
bash add.sh
```

该脚本会：
1. 使用 `data_new.txt` 中的数据对模型进行追加训练
2. 将新数据追加到 `data.txt` 并清空 `data_new.txt`
3. 启动对话模式

### 重新训练

如果需要完全重新训练（删除已有模型和词表）：

```bash
bash update.sh
```

## 环境依赖

- Python 3.8+
- PyTorch 2.0+
- tqdm
