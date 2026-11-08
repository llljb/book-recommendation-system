import os
import numpy as np
import pandas as pd
import json
import gzip
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.layers import (
    Input, Embedding, Dense, Flatten, Concatenate, Dropout, MultiHeadAttention
)
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


# =====================================
# 全局超参数配置
# =====================================
EMBEDDING_DIM = 64
DROPOUT_RATE = 0.2
EPOCHS = 5
BATCH_SIZE = 128


# =====================================
# 主类定义
# =====================================
class BookRecommender:
    def __init__(self, model_path="saved_models/recommendation_model"):
        self.model_path = model_path
        self.model = None
        self.user_to_idx = {}
        self.book_to_idx = {}
        self.idx_to_book = {}
        self.max_user_id = 0
        self.max_book_id = 0

    # --------------------------
    # 数据加载与预处理
    # --------------------------
    def load_and_preprocess_data(self, interactions_path, books_path, user_map_path, load_ratio=0.0001):
        print("🔹 正在加载并预处理数据...")

        # ===== 1. 用户映射表 =====
        user_map_df = pd.read_csv(user_map_path)
        user_map_df["user_id_csv"] = user_map_df["user_id_csv"].astype(str)
        user_map_df["user_id"] = user_map_df["user_id"].astype(str)

        user_to_idx_str = {str(row["user_id_csv"]): idx for idx, row in user_map_df.iterrows()}
        user_to_idx_int = {int(row["user_id_csv"]): idx for idx, row in user_map_df.iterrows()}
        self.user_to_idx = user_to_idx_str
        print(f"✅ 用户映射加载完成，共 {len(user_map_df):,} 个用户")

        # ===== 2. 书籍映射 =====
        book_to_idx = {}
        with gzip.open(books_path, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                book_data = json.loads(line)
                book_id = str(book_data.get("book_id"))  # ✅ 转为字符串
                if book_id not in book_to_idx:
                    book_to_idx[book_id] = i
        self.book_to_idx = book_to_idx
        self.idx_to_book = {v: k for k, v in book_to_idx.items()}
        print(f"✅ 书籍映射加载完成，共 {len(book_to_idx):,} 本书")

        # ===== 3. 限制读取行数 =====
        total_lines = 228648342
        limit = int(total_lines * load_ratio)
        print(f"✅ 交互数据总行数: {total_lines:,}, 将读取 {limit:,} 行")

        chunksize = 200000
        processed_chunks, rows_read = [], 0

        for chunk in pd.read_csv(interactions_path, chunksize=chunksize):
            print(f"🔸 正在处理数据块... 当前累计 {rows_read:,} 行")

            # 自动检测 user_id 类型
            sample_id = chunk["user_id"].iloc[0]
            if isinstance(sample_id, str):
                chunk["user_id"] = chunk["user_id"].astype(str)
                use_map = user_to_idx_str
            else:
                chunk["user_id"] = chunk["user_id"].astype(int)
                use_map = user_to_idx_int

            # ✅ 统一书籍 ID 类型为字符串
            chunk["book_id"] = chunk["book_id"].astype(str)

            # 匹配率检查
            user_match = chunk["user_id"].isin(use_map.keys()).mean() * 100
            book_match = chunk["book_id"].isin(self.book_to_idx.keys()).mean() * 100
            print(f"   ↳ 用户匹配率: {user_match:.2f}%, 书籍匹配率: {book_match:.2f}%")

            # 映射
            chunk["user_id"] = chunk["user_id"].map(use_map)
            chunk["book_id"] = chunk["book_id"].map(self.book_to_idx)

            # 删除无效样本
            before = len(chunk)
            chunk = chunk.dropna(subset=["user_id", "book_id"])
            after = len(chunk)
            print(f"✅ 当前块有效样本: {after:,} / {before:,}")

            if after > 0:
                processed_chunks.append(chunk)

            rows_read += len(chunk)
            if rows_read >= limit:
                break

        # 检查是否匹配到样本
        if not processed_chunks:
            raise ValueError("❌ 未匹配到任何样本，请检查 book_id 类型是否一致。")

        interactions_df = pd.concat(processed_chunks)
        print(f"✅ 实际加载样本数: {len(interactions_df):,}")

        # ===== 4. 划分训练 / 测试集 =====
        train_df, test_df = train_test_split(interactions_df, test_size=0.2, random_state=42)
        x_train = [train_df["user_id"].astype(int).values, train_df["book_id"].astype(int).values]
        y_train = train_df["rating"].values
        x_test = [test_df["user_id"].astype(int).values, test_df["book_id"].astype(int).values]
        y_test = test_df["rating"].values

        self.max_user_id = len(user_map_df)
        self.max_book_id = len(book_to_idx)
        print(f"✅ 用户数: {self.max_user_id:,}, 书籍数: {self.max_book_id:,}")

        return x_train, y_train, x_test, y_test

    # --------------------------
    # 构建注意力机制推荐模型
    # --------------------------
    def build_model_with_attention(self, num_heads=4):
        print("🔹 正在构建注意力机制推荐模型...")

        user_input = Input(shape=(1,), name="user_input")
        book_input = Input(shape=(1,), name="book_input")

        user_emb = Embedding(self.max_user_id, EMBEDDING_DIM, name="user_embedding")(user_input)
        book_emb = Embedding(self.max_book_id, EMBEDDING_DIM, name="book_embedding")(book_input)

        user_vec = Flatten()(user_emb)
        book_vec = Flatten()(book_emb)

        # 多头注意力机制
        query = tf.expand_dims(user_vec, axis=1)
        key = tf.expand_dims(book_vec, axis=1)
        value = tf.expand_dims(book_vec, axis=1)
        attn_output = MultiHeadAttention(num_heads=num_heads, key_dim=EMBEDDING_DIM)(query, value, key)
        attn_output = Flatten()(attn_output)

        concat = Concatenate()([user_vec, attn_output])
        hidden = Dense(128, activation="relu")(concat)
        hidden = Dropout(DROPOUT_RATE)(hidden)
        output = Dense(1)(hidden)

        self.model = Model(inputs=[user_input, book_input], outputs=output)
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss="mse", metrics=["mae"])
        print("✅ 模型构建完成！")

    # --------------------------
    # 模型训练（带进度条）
    # --------------------------
    def train(self, x_train, y_train, x_test, y_test, epochs=EPOCHS, batch_size=BATCH_SIZE):
        print("🔹 开始训练模型...")

        steps_per_epoch = int(len(x_train[0]) / batch_size)
        for epoch in range(epochs):
            print(f"\n📘 Epoch {epoch + 1}/{epochs}")
            progress = tqdm(range(steps_per_epoch), desc=f"Epoch {epoch + 1}/{epochs}", ncols=80)
            for step in progress:
                start, end = step * batch_size, (step + 1) * batch_size
                batch_x = [x[start:end] for x in x_train]
                batch_y = y_train[start:end]
                loss, mae = self.model.train_on_batch(batch_x, batch_y)
                progress.set_postfix({"Loss": f"{loss:.4f}", "MAE": f"{mae:.4f}"})

            val_loss, val_mae = self.model.evaluate(x_test, y_test, verbose=0)
            print(f"✅ 验证集 - Loss: {val_loss:.4f}, MAE: {val_mae:.4f}")

        print("🎯 训练完成！")
        self.save()

    # --------------------------
    def save(self):
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        self.model.save(self.model_path)
        print(f"✅ 模型已保存到：{self.model_path}")
