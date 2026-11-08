from project_config import ROOT, RUNTIME_DIR, MODEL_DIR as MODEL_DIR_CONFIG, data_path, model_path, runtime_path
# ============================================
# retrain_manager.py
# 增量式再训练模块（基于 user_id_map.csv 动态扩展）
# ============================================

import os
import json
import time
import gzip
import sqlite3
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model

# --------------------------------------------
# 配置路径
# --------------------------------------------
BASE_DIR = str(ROOT)
MODEL_DIR = str(MODEL_DIR_CONFIG)
MODEL_PATH = os.path.join(MODEL_DIR, "new_book_recommendation_model.h5")
BOOKS_PATH = data_path('goodreads_books.json.gz')
USER_MAP_PATH = data_path('user_id_map.csv')
FEEDBACK_DB = runtime_path("feedback.db")

os.makedirs(MODEL_DIR, exist_ok=True)

# --------------------------------------------
# Step 1. 从数据库加载反馈数据
# --------------------------------------------
def load_feedback_data():
    conn = sqlite3.connect(FEEDBACK_DB)
    df = pd.read_sql("SELECT user_id, book_id, feedback_type FROM user_interactions", conn)
    conn.close()
    return df


# --------------------------------------------
# Step 2. 预处理（将反馈类型映射为评分）
# --------------------------------------------
def preprocess_feedback(df: pd.DataFrame):
    rating_map = {"like": 5.0, "interested": 3.0, "dislike": 1.0}
    df["rating"] = df["feedback_type"].map(rating_map)
    df = df.dropna(subset=["user_id", "book_id", "rating"])
    return df


# --------------------------------------------
# Step 3. 动态扩展 user_id_map.csv
# --------------------------------------------
def update_user_id_map(df):
    # 读取旧映射
    if os.path.exists(USER_MAP_PATH):
        user_map = pd.read_csv(USER_MAP_PATH)
    else:
        user_map = pd.DataFrame(columns=["user_id_csv", "user_id"])

    existing_ids = set(user_map["user_id"])
    new_users = [u for u in df["user_id"].unique() if u not in existing_ids]

    if new_users:
        next_index = user_map["user_id_csv"].max() + 1 if len(user_map) else 0
        new_entries = pd.DataFrame({
            "user_id_csv": range(next_index, next_index + len(new_users)),
            "user_id": new_users
        })
        user_map = pd.concat([user_map, new_entries], ignore_index=True)
        user_map.to_csv(USER_MAP_PATH, index=False)
        print(f"✅ 已新增 {len(new_users)} 个新用户到 user_id_map.csv")

    return {r.user_id: int(r.user_id_csv) for r in user_map.itertuples()}


# --------------------------------------------
# Step 4. 简单的神经推荐模型
# --------------------------------------------
def build_simple_model(num_users, num_books, emb_dim=64):
    user_in = Input(shape=(1,))
    book_in = Input(shape=(1,))
    user_emb = Embedding(num_users, emb_dim)(user_in)
    book_emb = Embedding(num_books, emb_dim)(book_in)
    user_vec = Flatten()(user_emb)
    book_vec = Flatten()(book_emb)
    concat = Concatenate()([user_vec, book_vec])
    x = Dense(128, activation="relu")(concat)
    x = Dropout(0.3)(x)
    x = Dense(64, activation="relu")(x)
    out = Dense(1, activation="linear")(x)
    model = Model([user_in, book_in], out)
    model.compile(optimizer=Adam(1e-3), loss="mse")
    return model


# --------------------------------------------
# Step 5. 再训练流程（动态扩展用户）
# --------------------------------------------
def simple_retrain_model(df):
    # 1️⃣ 更新 user_id_map
    user_to_idx = update_user_id_map(df)

    # 2️⃣ book_id 动态映射（从书籍文件中提取）
    books = []
    with gzip.open(BOOKS_PATH, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 20000:
                break
            try:
                books.append(json.loads(line))
            except:
                continue
    book_df = pd.DataFrame(books)
    book_ids = list(book_df["book_id"].unique())
    book_to_idx = {b: i for i, b in enumerate(book_ids)}

    # 3️⃣ 构建训练数据
    x_users = np.array([user_to_idx[u] for u in df["user_id"] if u in user_to_idx])
    x_books = np.array([book_to_idx[b] for b in df["book_id"] if b in book_to_idx])
    y = df["rating"].astype(float).values

    # 4️⃣ 加载或新建模型
    if os.path.exists(MODEL_PATH):
        model = load_model(MODEL_PATH)
    else:
        model = build_simple_model(len(user_to_idx), len(book_to_idx))

    # 5️⃣ fine-tuning
    model.fit([x_users, x_books], y, epochs=3, batch_size=128, verbose=1)

    # 6️⃣ 保存模型
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(MODEL_DIR, f"retrained_model_{timestamp}.h5")
    model.save(save_path)

    return save_path, len(df)
