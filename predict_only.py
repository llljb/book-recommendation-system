from app.project_config import ROOT, RUNTIME_DIR, MODEL_DIR as MODEL_DIR_CONFIG, data_path, model_path, runtime_path
# predict_only.py
from book_recommender import BookRecommender

INTERACTIONS_PATH = data_path('goodreads_interactions.csv')
BOOKS_PATH = data_path('goodreads_books.json.gz')
USER_MAP_PATH = data_path('user_id_map.csv')
MODEL_SAVE_PATH = model_path('new_book_recommendation_model.h5')

if __name__ == "__main__":
    recommender = BookRecommender(model_path=MODEL_SAVE_PATH)
    # 这里仅加载少量交互样本用于建立必要映射（不会训练）
    recommender.load_and_preprocess_data(
        interactions_path=INTERACTIONS_PATH,
        books_path=BOOKS_PATH,
        user_map_path=USER_MAP_PATH,
        load_ratio=0.00001,   # 极小，仅用于建立映射
        sample_limit=200    # 或直接用 sample_limit 控制读取行数
    )

    # 指定你想推荐的用户（可传 interaction 中的数字 id，或 user_map 中的 string id）
    # 例如：test_user = 0  或  test_user = "8842281e1d134..."
    test_user = 0
    topk = 10
    recommender.recommend_books(user_id=test_user, top_k=topk)
