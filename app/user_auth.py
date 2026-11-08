from project_config import ROOT, RUNTIME_DIR, MODEL_DIR as MODEL_DIR_CONFIG, data_path, model_path, runtime_path
# =====================================
# user_auth.py
# 用户注册与登录模块 (SQLite)
# =====================================

import sqlite3
import hashlib
import uuid

DB_PATH = runtime_path('user_data.db')

# -----------------------------
# 数据库初始化
# -----------------------------
def init_user_db():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    username TEXT UNIQUE NOT NULL,
                    password TEXT NOT NULL,
                    user_id TEXT NOT NULL
                )''')
    conn.commit()
    conn.close()

# -----------------------------
# 密码加密函数
# -----------------------------
def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# -----------------------------
# 注册新用户
# -----------------------------
def register_user(username: str, password: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # 检查是否存在用户
    c.execute("SELECT * FROM users WHERE username=?", (username,))
    if c.fetchone():
        conn.close()
        return False, "❌ 用户名已存在，请更换！"

    user_id = str(uuid.uuid4())[:16]  # 生成唯一ID
    hashed_pw = hash_password(password)

    c.execute("INSERT INTO users (username, password, user_id) VALUES (?, ?, ?)",
              (username, hashed_pw, user_id))
    conn.commit()
    conn.close()
    return True, f"✅ 注册成功！您的用户ID为：{user_id}"

# -----------------------------
# 验证登录
# -----------------------------
def login_user(username: str, password: str):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    hashed_pw = hash_password(password)
    c.execute("SELECT user_id FROM users WHERE username=? AND password=?", (username, hashed_pw))
    result = c.fetchone()
    conn.close()
    if result:
        return True, result[0]  # 返回 user_id
    return False, "❌ 用户名或密码错误！"
