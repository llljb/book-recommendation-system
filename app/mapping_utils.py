# mapping_utils.py
import os
import json
import csv
from pathlib import Path
import sqlite3

def atomic_write_text(path, text):
    # 原子替换写入：先写 temp，再替换
    p = Path(path)
    tmp = p.with_suffix('.tmp')
    tmp.write_text(text, encoding='utf-8')
    tmp.replace(p)

def export_user_map_from_db(db_path, out_csv_path):
    """
    从 feedback.db 导出所有 distinct user_id，并按稳定顺序生成连续内部索引（0..N-1）。
    输出 CSV 包含两列：user_id, user_idx
    """
    conn = sqlite3.connect(db_path)
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT user_id FROM user_interactions")
    rows = [r[0] for r in cur.fetchall()]
    conn.close()

    # 稳定排序（你可以改为别的排序策略）
    rows = sorted([str(r) for r in rows])

    with open(out_csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['user_id', 'user_idx'])
        for i, uid in enumerate(rows):
            writer.writerow([uid, i])
    print(f"✅ 已导出 user_map CSV：{out_csv_path}（{len(rows)} users）")
    return out_csv_path

def load_user_map_from_csv(csv_path):
    """
    加载 CSV，返回映射 dict: {str(user_id): int(idx)}。
    如果 CSV 没有 user_idx 列或 idx 不连续，会自动重建连续索引（按文件顺序）。
    """
    p = Path(csv_path)
    if not p.exists():
        return {}
    with open(p, encoding='utf-8') as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)

    # 找列
    header = [h.strip() for h in header]
    if 'user_id' in header and 'user_idx' in header:
        id_i = header.index('user_id')
        idx_i = header.index('user_idx')
        entries = [(r[id_i].strip(), int(r[idx_i])) for r in rows]
        vals = [v for _, v in entries]
        # 验证连续性
        if set(vals) == set(range(len(vals))) and len(vals) == len(entries):
            return {str(k): int(v) for k, v in entries}
        else:
            # 重建：按文件顺序重编号
            new_map = {}
            for i, (k, _) in enumerate(entries):
                new_map[str(k)] = i
            return new_map
    else:
        # 如果只有一列 user_id，则按顺序编号
        if 'user_id' in header:
            id_i = header.index('user_id')
            new_map = {}
            for i, r in enumerate(rows):
                new_map[str(r[id_i].strip())] = i
            return new_map
        # 兜底：按每行第0列为 id
        new_map = {}
        for i, r in enumerate(rows):
            new_map[str(r[0].strip())] = i
        return new_map

def save_map_to_json(user_map, book_map, mapping_json_path):
    out = {'user_to_idx': user_map, 'book_to_idx': book_map}
    atomic_write_text(mapping_json_path, json.dumps(out, ensure_ascii=False, indent=2))
    print(f"✅ 已保存 mappings.json：{mapping_json_path}")

def update_mappings_using_csv(df, csv_path, mapping_json_path):
    """
    工作流程：
    1. 先从 CSV 加载 user_map（视为权威）
    2. 若 CSV 不存在，则尝试从 mapping_json_path 读取并规范化
    3. 对 df 中出现的新的 user_id/book_id，追加连续索引
    4. 保存 mappings.json（原子替换），并返回 user_to_idx, book_to_idx
    """
    # 1) 加载 user_map
    user_to_idx = {}
    book_to_idx = {}

    if os.path.exists(csv_path):
        user_to_idx = load_user_map_from_csv(csv_path)
        print(f"ℹ️ 从 CSV 加载 user_map，用户数：{len(user_to_idx)}")
    else:
        # 尝试从 mappings.json 加载并规范化
        if os.path.exists(mapping_json_path):
            raw = json.loads(open(mapping_json_path, encoding='utf-8').read())
            user_to_idx = {str(k): int(v) for k, v in raw.get('user_to_idx', {}).items()}
            # 规范化值为连续（按原 value 顺序）
            vals = sorted(set(user_to_idx.values()))
            if vals != list(range(len(vals))):
                items = sorted(user_to_idx.items(), key=lambda kv: int(kv[1]))
                user_to_idx = {k: i for i, (k, _) in enumerate(items)}
            print(f"ℹ️ 从 mappings.json 加载并规范化 user_map，用户数：{len(user_to_idx)}")
        else:
            print("ℹ️ 未检测到 CSV 与 mappings.json，先使用空映射（将由 df 补充）")

    # 处理 book 映射（如果你也有书籍 CSV，可以在此处加载；否则尝试读取 mapping_json）
    if os.path.exists(mapping_json_path):
        raw = json.loads(open(mapping_json_path, encoding='utf-8').read())
        book_to_idx = {str(k): int(v) for k, v in raw.get('book_to_idx', {}).items()}
        vals = sorted(set(book_to_idx.values()))
        if vals != list(range(len(vals))):
            items = sorted(book_to_idx.items(), key=lambda kv: int(kv[1]))
            book_to_idx = {k: i for i, (k, _) in enumerate(items)}
        print(f"ℹ️ 已加载书籍映射（若有），书籍数：{len(book_to_idx)}")

    # 2) 对 df 中的新 id 追加连续索引
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

    # 3) 保存 mappings.json 原子替换
    save_map_to_json(user_to_idx, book_to_idx, mapping_json_path)

    # 4) 可选：把 csv 同步更新为当前 authoritative（如果你希望 CSV 始终是权威）
    #    这样下一次加载就直接读 CSV。下面示例将覆盖 csv_path。
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['user_id', 'user_idx'])
        for k, v in sorted(user_to_idx.items(), key=lambda kv: kv[1]):
            writer.writerow([k, v])

    print(f"✅ mappings 已更新：users={len(user_to_idx)}, books={len(book_to_idx)}")
    return user_to_idx, book_to_idx
