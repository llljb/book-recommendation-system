from project_config import ROOT, RUNTIME_DIR, MODEL_DIR as MODEL_DIR_CONFIG, data_path, model_path, runtime_path
# ============================================
# app/feedback_exporter.py
# 导出用户反馈数据为标准 CSV，方便重新训练模型
# ============================================

import os
import sqlite3
import pandas as pd

DB_PATH = runtime_path('feedback.db')
EXPORT_PATH = runtime_path('feedback_export.csv')

def export_feedback_to_csv():
    """从 feedback.db 导出数据为与原始数据集相同格式"""
    if not os.path.exists(DB_PATH):
        raise FileNotFoundError(f"数据库不存在：{DB_PATH}")

    conn = sqlite3.connect(DB_PATH)
    try:
        df = pd.read_sql_query("SELECT * FROM user_interactions", conn)
    finally:
        conn.close()

    if df.empty:
        print("⚠️ 没有可导出的反馈数据。")
        return None

    # 映射 feedback_type → rating / is_read / is_reviewed
    df["is_read"] = 1
    df["is_reviewed"] = 0

    def map_rating(ftype):
        if ftype == "like":
            return 5
        elif ftype == "interested":
            return 3
        elif ftype == "dislike":
            return 1
        else:
            return 0

    df["rating"] = df["feedback_type"].apply(map_rating)

    # 只保留必要字段并按契约顺序导出
    export_df = df[["user_id", "book_id", "is_read", "rating", "is_reviewed"]]
    export_df.to_csv(EXPORT_PATH, index=False, encoding="utf-8")
    print(f"✅ 反馈数据已导出到：{EXPORT_PATH}（共 {len(export_df)} 条）")

    return export_df
