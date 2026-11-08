from project_config import ROOT, RUNTIME_DIR, MODEL_DIR as MODEL_DIR_CONFIG, data_path, model_path, runtime_path
import json
from pathlib import Path

MAPPING_PATH = model_path('mappings.json')  # 根据你的路径改

def normalize_mappings(path):
    p = Path(path)
    if not p.exists():
        print("mappings.json 不存在：", path)
        return
    data = json.loads(p.read_text(encoding='utf-8'))
    user_map = data.get('user_to_idx', {})
    book_map = data.get('book_to_idx', {})

    def remap(d, name):
        if not d:
            return d, False
        # 当前值集合与理想连续索引的比较
        keys = list(d.keys())
        vals = [int(v) for v in d.values()]
        max_v = max(vals)
        n_keys = len(keys)
        need_remap = not (set(vals) == set(range(n_keys)))
        print(f"{name}: len(keys)={n_keys}, max_val={max_v}, need_remap={need_remap}")
        if not need_remap:
            return d, False
        # 按原来的 value 顺序排序（可保留稳定性），再重新编号为 0..n-1
        items = sorted(d.items(), key=lambda kv: int(kv[1]))
        new_d = {k: i for i, (k, _) in enumerate(items)}
        return new_d, True

    new_user_map, ru = remap(user_map, 'user_to_idx')
    new_book_map, rb = remap(book_map, 'book_to_idx')

    if ru or rb:
        out = {'user_to_idx': new_user_map, 'book_to_idx': new_book_map}
        p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding='utf-8')
        print("已重写并规范化 mappings.json")
    else:
        print("mappings.json 已是规范形式（无需修改）")

if __name__ == "__main__":
    normalize_mappings(MAPPING_PATH)
