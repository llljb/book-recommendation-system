from project_config import ROOT, RUNTIME_DIR, MODEL_DIR as MODEL_DIR_CONFIG, data_path, model_path, runtime_path
# retrain_manager.py
# 再训练管理（使用 CSV 为权威映射，防止 embedding 缩小，稳健迁移）
import os
import json
import time
import sqlite3
import csv
from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Dropout
from tensorflow.keras.models import Model

# ------------------------- 配置（按需修改） -------------------------
BASE_DIR = str(ROOT)
MODEL_DIR = str(MODEL_DIR_CONFIG)
MODEL_PATH = os.path.join(MODEL_DIR, "new_book_recommendation_model.h5")
BOOKS_PATH = data_path('goodreads_books.json.gz')
USER_MAP_CSV = os.path.join(MODEL_DIR, "user_map.csv")   # 将 CSV 放到 saved_models 下
USER_MAP_PATH = data_path('user_id_map.csv')  # 旧路径仍保留用于兼容
FEEDBACK_DB = runtime_path("feedback.db")
MAPPING_PATH = os.path.join(MODEL_DIR, "mappings.json")

os.makedirs(MODEL_DIR, exist_ok=True)

# ------------------------- Step 1: 读取反馈 -------------------------
def load_feedback_data():
    conn = sqlite3.connect(FEEDBACK_DB)
    df = pd.read_sql("SELECT user_id, book_id, feedback_type FROM user_interactions", conn)
    conn.close()
    return df

# ------------------------- Step 2: 预处理反馈 -------------------------
def preprocess_feedback(df: pd.DataFrame):
    df = df.copy()
    rating_map = {"like": 5.0, "interested": 3.0, "dislike": 1.0}
    df = df.dropna(subset=["user_id", "book_id", "feedback_type"]).copy()
    df.loc[:, "rating"] = df.loc[:, "feedback_type"].map(rating_map)
    df = df.dropna(subset=["rating"]).copy()
    df.loc[:, "user_id"] = df.loc[:, "user_id"].astype(str)
    df.loc[:, "book_id"] = df.loc[:, "book_id"].astype(str)
    return df

# ------------------------- Step 3: 更新 user_id_map.csv（兼容旧逻辑） -------------------------
def update_user_id_map(db_path=FEEDBACK_DB, user_map_path=USER_MAP_PATH):
    conn = sqlite3.connect(db_path)
    users = pd.read_sql_query("SELECT DISTINCT user_id FROM user_interactions", conn)
    conn.close()

    if os.path.exists(user_map_path):
        df = pd.read_csv(user_map_path)
        df['user_id'] = df['user_id'].astype(str)
    else:
        df = pd.DataFrame(columns=["user_id_csv", "user_id"])

    existing = set(df["user_id"].astype(str))
    next_id = int(df["user_id_csv"].max()) + 1 if not df.empty else 0

    new_entries = []
    for uid in users["user_id"].astype(str):
        if uid not in existing:
            new_entries.append({"user_id_csv": next_id, "user_id": uid})
            next_id += 1

    if new_entries:
        df = pd.concat([df, pd.DataFrame(new_entries)], ignore_index=True)
        df.to_csv(user_map_path, index=False, encoding="utf-8-sig")
        print(f"✅ 新增 {len(new_entries)} 个新用户至 user_id_map.csv")
    else:
        print("ℹ️ 无需更新 user_id_map.csv")

    return df

# ------------------------- Mapping CSV 工具 -------------------------
def atomic_write_text(path, text):
    p = Path(path)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(p)

def export_user_map_from_db(db_path, out_csv_path):
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT user_id FROM user_interactions")
    rows = [r[0] for r in cur.fetchall()]
    conn.close()
    rows = sorted([str(r) for r in rows])
    with open(out_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['user_id', 'user_idx'])
        for i, uid in enumerate(rows):
            writer.writerow([uid, i])
    print(f"✅ 已导出 user_map CSV：{out_csv_path}（{len(rows)} users）")
    return out_csv_path

def load_user_map_from_csv(csv_path):
    p = Path(csv_path)
    if not p.exists():
        return {}
    with open(p, encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    header = [h.strip() for h in header]
    if 'user_id' in header and 'user_idx' in header:
        id_i = header.index('user_id')
        idx_i = header.index('user_idx')
        entries = [(r[id_i].strip(), int(r[idx_i])) for r in rows]
        vals = [v for _, v in entries]
        if set(vals) == set(range(len(vals))) and len(vals) == len(entries):
            return {str(k): int(v) for k, v in entries}
        else:
            new_map = {}
            for i, (k, _) in enumerate(entries):
                new_map[str(k)] = i
            return new_map
    else:
        if 'user_id' in header:
            id_i = header.index('user_id')
            new_map = {}
            for i, r in enumerate(rows):
                new_map[str(r[id_i].strip())] = i
            return new_map
        new_map = {}
        for i, r in enumerate(rows):
            new_map[str(r[0].strip())] = i
        return new_map

def save_map_to_json(user_map, book_map, mapping_json_path):
    out = {'user_to_idx': user_map, 'book_to_idx': book_map}
    atomic_write_text(mapping_json_path, json.dumps(out, ensure_ascii=False, indent=2))
    print(f"✅ 已保存 mappings.json：{mapping_json_path}")

def normalize_mappings():
    """
    规范已有 mappings.json（将值重建为 0..N-1 紧凑索引），用于手动修复。
    """
    p = MAPPING_PATH
    if not os.path.exists(p):
        print("mappings.json 不存在：", p)
        return
    data = json.loads(open(p, encoding='utf-8').read())
    user_map = data.get('user_to_idx', {})
    book_map = data.get('book_to_idx', {})

    def remap(d, name):
        if not d:
            return d, False
        keys = list(d.keys())
        vals = [int(v) for v in d.values()]
        max_v = max(vals) if vals else None
        n_keys = len(keys)
        need_remap = not (set(vals) == set(range(n_keys)))
        print(f"{name}: len(keys)={n_keys}, max_val={max_v}, need_remap={need_remap}")
        if not need_remap:
            return d, False
        items = sorted(d.items(), key=lambda kv: int(kv[1]))
        new_d = {k: i for i, (k, _) in enumerate(items)}
        return new_d, True

    new_user_map, ru = remap(user_map, 'user_to_idx')
    new_book_map, rb = remap(book_map, 'book_to_idx')

    if ru or rb:
        out = {'user_to_idx': new_user_map, 'book_to_idx': new_book_map}
        open(p, 'w', encoding='utf-8').write(json.dumps(out, ensure_ascii=False, indent=2))
        print("已重写并规范化 mappings.json")
    else:
        print("mappings.json 已是规范形式（无需修改）")

# ------------------------- update_mappings_using_csv -------------------------
def update_mappings_using_csv(df, csv_path=USER_MAP_CSV, mapping_json_path=MAPPING_PATH):
    """
    流程：
    1. 优先从 CSV 加载 user_map（视为权威）
    2. 若 CSV 不存在则尝试从 mappings.json 读取并规范化
    3. 对 df 中新出现的 id 追加连续索引
    4. 保存 mappings.json（原子替换）并覆盖 CSV（保持同步）
    """
    user_to_idx = {}
    book_to_idx = {}

    if os.path.exists(csv_path):
        user_to_idx = load_user_map_from_csv(csv_path)
        print(f"ℹ️ 从 CSV 加载 user_map，用户数：{len(user_to_idx)}")
    else:
        if os.path.exists(mapping_json_path):
            raw = json.loads(open(mapping_json_path, encoding='utf-8').read())
            user_to_idx = {str(k): int(v) for k, v in raw.get('user_to_idx', {}).items()}
            vals = sorted(set(user_to_idx.values()))
            if vals != list(range(len(vals))):
                items = sorted(user_to_idx.items(), key=lambda kv: int(kv[1]))
                user_to_idx = {k: i for i, (k, _) in enumerate(items)}
            print(f"ℹ️ 从 mappings.json 加载并规范化 user_map，用户数：{len(user_to_idx)}")
        else:
            print("ℹ️ 未检测到 CSV 与 mappings.json，先使用空映射（将由 df 补充）")

    if os.path.exists(mapping_json_path):
        raw = json.loads(open(mapping_json_path, encoding='utf-8').read())
        book_to_idx = {str(k): int(v) for k, v in raw.get('book_to_idx', {}).items()}
        vals = sorted(set(book_to_idx.values()))
        if vals != list(range(len(vals))):
            items = sorted(book_to_idx.items(), key=lambda kv: int(kv[1]))
            book_to_idx = {k: i for i, (k, _) in enumerate(items)}
        print(f"ℹ️ 已加载书籍映射（若有），书籍数：{len(book_to_idx)}")

    cur_user_len = len(user_to_idx)
    for u in df["user_id"].astype(str).unique():
        if u not in user_to_idx:
            user_to_idx[u] = cur_user_len
            cur_user_len += 1

    cur_book_len = len(book_to_idx)
    for b in df["book_id"].astype(str).unique():
        if b not in book_to_idx:
            book_to_idx[b] = cur_book_len
            cur_book_len += 1

    save_map_to_json(user_to_idx, book_to_idx, mapping_json_path)

    # 同步覆盖 CSV（保持 CSV 与 mappings.json 一致）
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['user_id', 'user_idx'])
        for k, v in sorted(user_to_idx.items(), key=lambda kv: kv[1]):
            writer.writerow([k, v])

    print(f"✅ mappings 已更新：users={len(user_to_idx)}, books={len(book_to_idx)}")
    return user_to_idx, book_to_idx

# ------------------------- 构建模型与迁移逻辑 -------------------------
def build_simple_model(num_users, num_books, emb_dim=64):
    user_in = Input(shape=(1,), name='user_in')
    book_in = Input(shape=(1,), name='book_in')
    user_emb = Embedding(num_users, emb_dim, name='user_emb')(user_in)
    book_emb = Embedding(num_books, emb_dim, name='book_emb')(book_in)
    user_vec = Flatten(name='user_flat')(user_emb)
    book_vec = Flatten(name='book_flat')(book_emb)
    concat = Concatenate(name='concat')([user_vec, book_vec])
    x = Dense(128, activation="relu", name='fc1')(concat)
    x = Dropout(0.3, name='dropout')(x)
    x = Dense(64, activation="relu", name='fc2')(x)
    out = Dense(1, activation="linear", name='pred')(x)
    model = Model([user_in, book_in], out)
    model.compile(optimizer=Adam(1e-3), loss="mse")
    return model

def extract_embedding_weights(old_model):
    emb_layers = [l for l in old_model.layers if l.__class__.__name__ == 'Embedding']
    if not emb_layers:
        return None, None
    user_layer = None
    book_layer = None
    for l in emb_layers:
        name = l.name.lower()
        if 'user' in name:
            user_layer = l
        elif 'book' in name or 'item' in name:
            book_layer = l
    if user_layer is None:
        user_layer = emb_layers[0]
    if book_layer is None and len(emb_layers) > 1:
        book_layer = emb_layers[1]
    user_w = user_layer.get_weights()[0] if user_layer is not None else None
    book_w = book_layer.get_weights()[0] if book_layer is not None else None
    return user_w, book_w

def build_model_with_transfer(old_model, new_num_users, new_num_books):
    old_user_w, old_book_w = extract_embedding_weights(old_model)
    if old_user_w is not None:
        emb_dim = old_user_w.shape[1]
    elif old_book_w is not None:
        emb_dim = old_book_w.shape[1]
    else:
        emb_dim = 64

    print(f"🧩 迁移：目标 user_count={new_num_users}, book_count={new_num_books}, emb_dim={emb_dim}")
    new_model = build_simple_model(new_num_users, new_num_books, emb_dim=emb_dim)

    try:
        if old_user_w is not None:
            new_user_layer = new_model.get_layer('user_emb')
            new_user_w = new_user_layer.get_weights()[0]
            copy_n = min(old_user_w.shape[0], new_user_w.shape[0])
            copy_dim = min(old_user_w.shape[1], new_user_w.shape[1])
            new_user_w[:copy_n, :copy_dim] = old_user_w[:copy_n, :copy_dim]
            new_user_layer.set_weights([new_user_w])
            print(f"✅ 已迁移 user_emb：从 {old_user_w.shape} 到 {new_user_w.shape} (copy_n={copy_n}, copy_dim={copy_dim})")
        else:
            print("⚠️ 未检测到旧的 user embedding，跳过迁移")
    except Exception as e:
        print(f"⚠️ 无法迁移 user_emb：{e}")

    try:
        if old_book_w is not None:
            new_book_layer = new_model.get_layer('book_emb')
            new_book_w = new_book_layer.get_weights()[0]
            copy_n = min(old_book_w.shape[0], new_book_w.shape[0])
            copy_dim = min(old_book_w.shape[1], new_book_w.shape[1])
            new_book_w[:copy_n, :copy_dim] = old_book_w[:copy_n, :copy_dim]
            new_book_layer.set_weights([new_book_w])
            print(f"✅ 已迁移 book_emb：从 {old_book_w.shape} 到 {new_book_w.shape} (copy_n={copy_n}, copy_dim={copy_dim})")
        else:
            print("⚠️ 未检测到旧的 book embedding，跳过迁移")
    except Exception as e:
        print(f"⚠️ 无法迁移 book_emb：{e}")

    for lname in ['fc1', 'fc2', 'pred']:
        try:
            old_layer = old_model.get_layer(lname)
            new_layer = new_model.get_layer(lname)
            old_w = old_layer.get_weights()
            new_w = new_layer.get_weights()
            if len(old_w) == len(new_w) and all([ow.shape == nw.shape for ow, nw in zip(old_w, new_w)]):
                new_layer.set_weights(old_w)
                print(f"✅ 已迁移层 {lname}")
        except Exception:
            continue

    new_model.compile(optimizer=Adam(1e-3), loss='mse')
    return new_model

# ------------------------- 最终再训练主流程（使用 CSV 权威 & 不缩小 embedding） -------------------------
def simple_retrain_model(df, use_csv=True):
    # 更新映射：优先使用 CSV
    if use_csv:
        user_to_idx, book_to_idx = update_mappings_using_csv(df, csv_path=USER_MAP_CSV, mapping_json_path=MAPPING_PATH)
    else:
        user_to_idx, book_to_idx = update_mappings_using_csv(df, csv_path=USER_MAP_CSV, mapping_json_path=MAPPING_PATH)

    x_users = np.array([user_to_idx[str(u)] for u in df["user_id"].astype(str)])
    x_books = np.array([book_to_idx[str(b)] for b in df["book_id"].astype(str)])
    y = df["rating"].astype(float).values

    print(f"ℹ️ 再训练样本数：{len(y)}, 本次 df 不同用户数:{len(set(x_users))}, 本次 df 不同书籍数:{len(set(x_books))}")
    print(f"ℹ️ 当前映射长度：len(user_to_idx)={len(user_to_idx)}, len(book_to_idx)={len(book_to_idx)}")

    desired_num_users = len(user_to_idx)
    desired_num_books = len(book_to_idx)

    if os.path.exists(MODEL_PATH):
        try:
            old_model = load_model(MODEL_PATH)
            old_user_rows = None
            old_book_rows = None
            try:
                old_user_rows = old_model.get_layer('user_emb').get_weights()[0].shape[0]
            except Exception:
                old_user_w, old_book_w = extract_embedding_weights(old_model)
                old_user_rows = old_user_w.shape[0] if old_user_w is not None else None

            try:
                old_book_rows = old_model.get_layer('book_emb').get_weights()[0].shape[0]
            except Exception:
                if 'old_book_w' not in locals():
                    _, old_book_w = extract_embedding_weights(old_model)
                old_book_rows = old_book_w.shape[0] if old_book_w is not None else None

            print(f"ℹ️ 旧模型 embedding 行数：old_user_rows={old_user_rows}, old_book_rows={old_book_rows}")

            new_num_users = desired_num_users if old_user_rows is None else max(desired_num_users, int(old_user_rows))
            new_num_books = desired_num_books if old_book_rows is None else max(desired_num_books, int(old_book_rows))

            model = build_model_with_transfer(old_model, new_num_users, new_num_books)
            print(f"ℹ️ 使用旧模型权重构建并扩容/保持至新大小。new_user_count={new_num_users}, new_book_count={new_num_books}")
        except Exception as e:
            print(f"⚠️ 加载旧模型或迁移权重失败，改为从头训练：{e}")
            model = build_simple_model(desired_num_users, desired_num_books)
    else:
        model = build_simple_model(desired_num_users, desired_num_books)

    try:
        user_emb_w = model.get_layer('user_emb').get_weights()[0]
        book_emb_w = model.get_layer('book_emb').get_weights()[0]
        print(f"🧩 训练前 embedding 尺寸：user={user_emb_w.shape}, book={book_emb_w.shape}")
    except Exception as e:
        print(f"⚠️ 无法读取新模型 embedding：{e}")

    model.fit([x_users, x_books], y, epochs=3, batch_size=128, verbose=1)

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(MODEL_DIR, f"retrained_model_{timestamp}.h5")
    model.save(save_path)

    update_user_id_map(FEEDBACK_DB, USER_MAP_PATH)

    print(f"✅ 模型再训练完成，保存路径：{save_path}")
    return save_path, len(df)

# ------------------------- 如果直接运行脚本则执行 normalize_mappings -------------------------
if __name__ == "__main__":
    print("Notice: 运行 retrain_manager.py 作为脚本将执行 normalize_mappings()（不会触发训练）。")
    normalize_mappings()
