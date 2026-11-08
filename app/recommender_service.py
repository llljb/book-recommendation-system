# ============================================
# recommender_service.py
# 图书推荐核心服务模块（动态索引检测 + 安全回退）
# ============================================

import os
import gzip
import json
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Embedding

class RecommenderService:
    def __init__(self, model_path, books_path, user_map_path, top_k=10):
        self.model_path = model_path
        self.books_path = books_path
        self.user_map_path = user_map_path
        self.top_k = top_k

        # 备用模型
        self.default_model_path = os.path.join(
            os.path.dirname(model_path),
            "new_book_recommendation_model.h5"
        )

        self.model = None
        self.books_df = None
        self.user_to_idx = {}
        self.book_to_idx = {}
        self.idx_to_book = {}
        self.user_emb_size = 0
        self.book_emb_size = 0

        self._load_all()

    # ----------------------------------------
    def _load_all(self):
        self._load_model()
        self._load_books()
        self._load_user_map()
        self._build_book_index()

    # ----------------------------------------
    def _load_model(self):
        if os.path.exists(self.model_path):
            self.model = load_model(self.model_path)
            print(f"✅ 已加载模型：{self.model_path}")
        elif os.path.exists(self.default_model_path):
            self.model = load_model(self.default_model_path)
            print(f"⚠️ 未找到新模型，已加载默认模型：{self.default_model_path}")
        else:
            raise FileNotFoundError(f"❌ 未找到任何模型：{self.model_path}")

        # 自动检测 embedding 尺寸
        self.user_emb_size, self.book_emb_size = self._detect_embedding_size()

        print(f"🧩 检测到 embedding 尺寸：user={self.user_emb_size}, book={self.book_emb_size}")

    def _detect_embedding_size(self):
        """智能检测模型 embedding 层大小"""
        user_size = 0
        book_size = 0
        for layer in self.model.layers:
            if isinstance(layer, Embedding):
                if user_size == 0:
                    user_size = layer.input_dim
                else:
                    book_size = layer.input_dim
        return user_size, book_size

    # ----------------------------------------
    def _load_books(self):
        books = []
        with gzip.open(self.books_path, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 30000:
                    break
                try:
                    books.append(json.loads(line))
                except:
                    continue
        self.books_df = pd.DataFrame(books)
        if "book_id" not in self.books_df.columns:
            self.books_df["book_id"] = [f"book_{i}" for i in range(len(self.books_df))]
        print(f"📚 已加载书籍 {len(self.books_df)} 条")

    # ----------------------------------------
    def _load_user_map(self):
        import pandas as pd

        if not os.path.exists(self.user_map_path):
            print(f"⚠️ 用户映射文件 {self.user_map_path} 不存在，将创建空映射。")
            self.user_to_idx = {}
            return

        df = pd.read_csv(self.user_map_path, dtype=str)

        # 兼容旧格式和新格式
        if {"user_id", "user_id_csv"}.issubset(df.columns):
            # 旧版格式（user_id_map.csv）
            df.rename(columns={"user_id_csv": "user_idx"}, inplace=True)
        elif {"user_id", "user_idx"}.issubset(df.columns):
            # 新版格式（user_map.csv）
            pass
        else:
            raise ValueError("❌ user_map.csv 缺少必要列（user_id, user_idx）或（user_id, user_id_csv）")

        df["user_idx"] = df["user_idx"].astype(int)
        self.user_to_idx = dict(zip(df["user_id"].astype(str), df["user_idx"]))
        print(f"✅ 已加载用户映射，共 {len(self.user_to_idx)} 个用户。")

    # ----------------------------------------
    def _build_book_index(self):
        book_ids = list(self.books_df["book_id"].unique())
        self.book_to_idx = {b: i for i, b in enumerate(book_ids)}
        self.idx_to_book = {i: b for b, i in self.book_to_idx.items()}
        print(f"📖 已建立书籍索引 {len(self.book_to_idx)} 本")

    # ----------------------------------------
    def recommend(self, user_id, top_k=None):
        if top_k is None:
            top_k = self.top_k

        if user_id not in self.user_to_idx:
            print(f"⚠️ 用户 {user_id} 不在映射表中 → 热门推荐")
            return self._recommend_popular(top_k)

        user_idx = self.user_to_idx[user_id]

        # ✅ 安全检查索引范围
        if user_idx >= self.user_emb_size or self.user_emb_size == 0:
            print(f"⚠️ 用户索引 {user_idx} 超出模型范围（最大 {self.user_emb_size}） → 热门推荐")
            return self._recommend_popular(top_k)

        book_indices = np.arange(len(self.book_to_idx))
        user_input = np.full_like(book_indices, user_idx)

        try:
            preds = self.model.predict([user_input, book_indices], verbose=0).flatten()
        except Exception as e:
            print(f"❌ 模型预测失败：{e}")
            return self._recommend_popular(top_k)

        top_indices = np.argsort(preds)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            book_id = self.idx_to_book[idx]
            book_row = self.books_df[self.books_df["book_id"] == book_id]
            if book_row.empty:
                continue
            title = book_row.iloc[0].get("title", "未知标题")
            image_url = book_row.iloc[0].get("image_url", "https://via.placeholder.com/120x160.png?text=No+Cover")
            results.append({
                "book_id": book_id,
                "title": title,
                "image_url": image_url
            })
        return results

    # ----------------------------------------
    def _recommend_popular(self, top_k=10):
        """本地热门推荐（冷启动回退）"""
        if "ratings_count" in self.books_df.columns:
            df = self.books_df.copy()
            df["ratings_count"] = pd.to_numeric(df["ratings_count"], errors="coerce").fillna(0)
            df = df.sort_values("ratings_count", ascending=False).head(top_k)
        else:
            df = self.books_df.head(top_k)

        return [
            {
                "book_id": r.get("book_id", f"book_{i}"),
                "title": r.get("title", "未知标题"),
                "image_url": r.get("image_url", "https://via.placeholder.com/120x160.png?text=No+Cover")
            }
            for i, r in df.iterrows()
        ]
