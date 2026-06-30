#!/bin/bash

# 对模型进行追加训练
python3 train_llm.py \
  --data_path ./data_new.txt \
  --save_model ./my_model.pt \
  --save_vocab ./vocab.json \
  --vocab_path ./vocab.json \
  --epochs 2 \
  --batch_size 8 \
  --seq_len 32 \
  --lr 0.0001

# 追加 data_new.txt 到 data.txt 并清空
DATA_FILE="data.txt"
NEW_DATA_FILE="data_new.txt"

if [ -f "$NEW_DATA_FILE" ] && [ -s "$NEW_DATA_FILE" ]; then
    cat "$NEW_DATA_FILE" >> "$DATA_FILE"
    > "$NEW_DATA_FILE"
    echo "已将 $NEW_DATA_FILE 的内容追加到 $DATA_FILE，并清空"
elif [ -f "$NEW_DATA_FILE" ]; then
    echo "提示：$NEW_DATA_FILE 为空，无需操作"
else
    echo "错误：$NEW_DATA_FILE 不存在"
    exit 1
fi

# 启动对话
bash chat.sh