from project_config import ROOT, RUNTIME_DIR, MODEL_DIR as MODEL_DIR_CONFIG, data_path, model_path, runtime_path
# ============================================
# app/feedback_manager.py
# 用户反馈数据管理模块（带自动修复表结构 + 自动扩展 user_id_map.csv）
# ============================================

import os
import sqlite3
from datetime import datetime
import pandas as pd

# -------------------------
# 基础路径配置
# -------------------------
BASE_DIR = str(RUNTIME_DIR)
DB_PATH = os.path.join(BASE_DIR, "feedback.db")
USER_MAP_PATH = data_path('user_id_map.csv')  # ✅ 用户映射表路径（与你的 app 一致）


# -------------------------
# 初始化反馈数据库
# -------------------------
def init_feedback_db():
    """初始化反馈数据库，并自动修复缺失字段"""
    os.makedirs(BASE_DIR, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 创建表（如不存在）
    cur.execute('''
        CREATE TABLE IF NOT EXISTS user_interactions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id TEXT,
            book_id TEXT,
            is_read INTEGER DEFAULT 0,
            rating REAL DEFAULT NULL,
            is_reviewed INTEGER DEFAULT 0,
            feedback_type TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')

    # 自动补缺失列
    cur.execute("PRAGMA table_info(user_interactions)")
    existing_cols = [col[1] for col in cur.fetchall()]
    if "feedback_type" not in existing_cols:
        cur.execute("ALTER TABLE user_interactions ADD COLUMN feedback_type TEXT DEFAULT NULL")
    if "created_at" not in existing_cols:
        cur.execute("ALTER TABLE user_interactions ADD COLUMN created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP")

    conn.commit()
    conn.close()


# -------------------------
# 扩展 user_id_map.csv
# -------------------------
def ensure_user_in_map(user_id):
    """若 user_id 不在 user_id_map.csv 中，则自动追加"""
    try:
        if not os.path.exists(USER_MAP_PATH):
            df = pd.DataFrame(columns=["user_id_csv", "user_id"])
        else:
            df = pd.read_csv(USER_MAP_PATH)

        if user_id not in df["user_id"].astype(str).values:
            new_idx = int(df["user_id_csv"].max() + 1) if not df.empty else 0
            new_row = pd.DataFrame([{"user_id_csv": new_idx, "user_id": user_id}])
            df = pd.concat([df, new_row], ignore_index=True)
            df.to_csv(USER_MAP_PATH, index=False, encoding="utf-8-sig")
            print(f"✅ 已新增新用户到 user_id_map.csv: {user_id}（index={new_idx}）")
        else:
            print(f"ℹ️ 用户 {user_id} 已存在 user_id_map.csv。")

    except Exception as e:
        print(f"⚠️ 写入 user_id_map.csv 时出错: {e}")


# -------------------------
# 保存用户反馈
# -------------------------
def save_feedback(user_id, book_id, feedback_type):
    """保存用户反馈数据，并自动更新 user_id_map"""
    init_feedback_db()
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # 根据反馈类型映射行为
    if feedback_type == "like":
        is_read, rating, is_reviewed = 1, 5, 1
    elif feedback_type == "interested":
        is_read, rating, is_reviewed = 0, 4, 0
    else:  # dislike
        is_read, rating, is_reviewed = 1, 2, 0

    cur.execute('''
        INSERT INTO user_interactions (user_id, book_id, is_read, rating, is_reviewed, feedback_type, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, book_id, is_read, rating, is_reviewed, feedback_type, datetime.now()))
    conn.commit()
    conn.close()

    # ✅ 同步更新 user_id_map
    ensure_user_in_map(user_id)


# -------------------------
# 获取用户反馈摘要
# -------------------------
def get_user_feedback_summary(user_id):
    """返回用户反馈统计信息"""
    init_feedback_db()
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query("SELECT * FROM user_interactions WHERE user_id = ?", conn, params=(user_id,))
    conn.close()

    if df.empty:
        return None

    # 确保关键列存在
    for col in ["feedback_type", "rating"]:
        if col not in df.columns:
            df[col] = None

    summary = {
        "total": len(df),
        "likes": (df["feedback_type"] == "like").sum(),
        "interests": (df["feedback_type"] == "interested").sum(),
        "dislikes": (df["feedback_type"] == "dislike").sum(),
        "avg_rating": round(df["rating"].mean(), 2) if not df["rating"].isna().all() else None,
        "books": df[["book_id", "feedback_type", "created_at"]]
        .sort_values("created_at", ascending=False)
        .to_dict(orient="records")
    }
    return summary
