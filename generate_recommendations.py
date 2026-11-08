from app.project_config import ROOT, RUNTIME_DIR, MODEL_DIR as MODEL_DIR_CONFIG, data_path, model_path, runtime_path
# =====================================
# generate_recommendations.py
# 使用已训练模型生成推荐结果（低内存版）
# =====================================

import os
import gzip
import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model

# -----------------------------
# 路径配置（按实际情况修改）
# -----------------------------
MODEL_PATH = model_path('new_book_recommendation_model.h5')
BOOKS_PATH = data_path('goodreads_books.json.gz')
USER_MAP_PATH = data_path('user_id_map.csv')

# -----------------------------
# 加载用户映射与书籍信息
# -----------------------------
print("🔹 正在加载映射与元数据...")

user_map = pd.read_csv(USER_MAP_PATH)
user_to_idx = dict(zip(user_map["user_id"], range(len(user_map))))
idx_to_user = {v: k for k, v in user_to_idx.items()}

# 书籍数据（仅取关键字段）
books = []
with gzip.open(BOOKS_PATH, "rt", encoding="utf-8") as f:
    for i, line in enumerate(f):
        book = json.loads(line)
        books.append({
            "book_id": book.get("book_id"),
            "title": book.get("title", "未知标题"),
            "image_url": book.get("image_url", "")
        })
books_df = pd.DataFrame(books)
book_to_idx = dict(zip(books_df["book_id"], range(len(books_df))))
idx_to_book = {v: k for k, v in book_to_idx.items()}

print(f"✅ 用户数: {len(user_to_idx):,}, 书籍数: {len(book_to_idx):,}")

# -----------------------------
# 加载模型
# -----------------------------
print(f"🔹 正在加载模型：{MODEL_PATH}")
model = load_model(MODEL_PATH)
print("✅ 模型加载完成！")

# -----------------------------
# 生成推荐（低内存版）
# -----------------------------
def recommend_books(user_id, top_k=10, batch_size=2000):
    import heapq

    if user_id not in user_to_idx:
        raise ValueError(f"❌ 用户 {user_id} 不存在于映射表中。")

    user_idx = user_to_idx[user_id]
    n_books = len(book_to_idx)
    print(f"\n🚀 为用户 {user_id} 生成推荐，共 {n_books:,} 本书...")

    top_heap = []  # 小顶堆
    for start in range(0, n_books, batch_size):
        end = min(start + batch_size, n_books)
        books_idx = np.arange(start, end, dtype=np.int32)
        user_array = np.full_like(books_idx, user_idx, dtype=np.int32)

        preds = model.predict([user_array, books_idx], verbose=0).reshape(-1)

        # 保留 top_k
        for score, bidx in zip(preds, books_idx):
            if len(top_heap) < top_k:
                heapq.heappush(top_heap, (score, bidx))
            else:
                if score > top_heap[0][0]:
                    heapq.heapreplace(top_heap, (score, bidx))

        if (start // batch_size) % 10 == 0:
            print(f"   进度: {end:,}/{n_books:,}")

    top_heap.sort(reverse=True)
    top_indices = [bidx for _, bidx in top_heap]
    recommended_books = [idx_to_book[int(idx)] for idx in top_indices]

    print("\n📚 Top-10 推荐结果：")
    for rank, bid in enumerate(recommended_books, 1):
        row = books_df.loc[books_df["book_id"] == bid].iloc[0]
        print(f"Top {rank}: {row['title']}")
        print(f"   封面: {row['image_url']}")

    return recommended_books


# -----------------------------
# 主程序执行
# -----------------------------
if __name__ == "__main__":
    # 随便挑一个用户 ID
    test_user_id = user_map.iloc[0]["user_id"]
    recommend_books(test_user_id, top_k=10)
