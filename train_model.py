from app.project_config import ROOT, RUNTIME_DIR, MODEL_DIR as MODEL_DIR_CONFIG, data_path, model_path, runtime_path
# =====================================
# train_model.py
# 用于训练 BookRecommender 模型
# =====================================

from book_recommender import BookRecommender

# -----------------------------
# 路径配置（请根据你的实际路径修改）
# -----------------------------
INTERACTIONS_PATH = data_path('goodreads_interactions.csv')
BOOKS_PATH = data_path('goodreads_books.json.gz')
USER_MAP_PATH = data_path('user_id_map.csv')
MODEL_SAVE_PATH = model_path('new_book_recommendation_model.h5')

# -----------------------------
# 主程序
# -----------------------------
if __name__ == "__main__":
    print("🚀 开始图书推荐模型训练流程...")

    recommender = BookRecommender(model_path=MODEL_SAVE_PATH)

    x_train, y_train, x_test, y_test = recommender.load_and_preprocess_data(
        interactions_path=INTERACTIONS_PATH,
        books_path=BOOKS_PATH,
        user_map_path=USER_MAP_PATH,
        load_ratio=0.001
    )

    recommender.build_model_with_attention()
    recommender.train(
        x_train, y_train,
        x_test, y_test,
        epochs=5,
        batch_size=256
    )

    # 测试推荐结果
    test_user_id = list(recommender.user_to_idx.keys())[0]
    print(f"\n为用户 {test_user_id} 生成 Top-10 推荐：")
    recommender.recommend_books(user_id=test_user_id, top_k=10)

    print("\n🔍 可视化书籍嵌入...")
    recommender.visualize_embeddings("embedding")

    print("\n✅ 模型训练与测试流程结束！模型已保存。")
