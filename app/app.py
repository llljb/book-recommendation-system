from project_config import ROOT, RUNTIME_DIR, MODEL_DIR as MODEL_DIR_CONFIG, data_path, model_path, runtime_path
# app.py
# Streamlit 应用（兼容旧版 Streamlit + 修复列嵌套问题）

import os
import gzip
import json
import time
import csv
from pathlib import Path

import streamlit as st
import pandas as pd
import tensorflow as tf

from user_auth import init_user_db, register_user, login_user
from recommender_service import RecommenderService
from feedback_manager import init_feedback_db, save_feedback, get_user_feedback_summary
from feedback_exporter import export_feedback_to_csv
from retrain_manager import (
    load_feedback_data,
    preprocess_feedback,
    simple_retrain_model,
    export_user_map_from_db,
)

# ------------------------- 1. 配置路径 -------------------------
BASE_DIR = str(ROOT)
MODEL_DIR = str(MODEL_DIR_CONFIG)


def get_latest_model_path(model_dir, fallback_name="new_book_recommendation_model.h5"):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    models = [f for f in os.listdir(model_dir) if f.endswith('.h5')]
    if not models:
        return os.path.join(model_dir, fallback_name)
    models = sorted(models)
    return os.path.join(model_dir, models[-1])


MODEL_PATH = get_latest_model_path(MODEL_DIR)
BOOKS_PATH = data_path('goodreads_books.json.gz')

USER_MAP_CSV = os.path.join(MODEL_DIR, "user_map.csv")
LEGACY_USER_MAP_PATH = data_path('user_id_map.csv')
DB_PATH = runtime_path("feedback.db")

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

init_user_db()
init_feedback_db()

st.set_page_config(page_title="图书推荐系统", layout="wide", page_icon="📚")


# ------------------------- 2. 核心辅助函数 -------------------------

def ensure_user_map_csv(user_map_csv, legacy_user_map_path, db_path):
    p = Path(user_map_csv)
    if p.exists():
        return
    try:
        lp = Path(legacy_user_map_path)
        if lp.exists():
            df_legacy = pd.read_csv(lp, dtype=str)
            if 'user_id_csv' in df_legacy.columns and 'user_id' in df_legacy.columns:
                out = df_legacy[['user_id', 'user_id_csv']].rename(columns={'user_id_csv': 'user_idx'})
                out.to_csv(user_map_csv, index=False, encoding='utf-8-sig')
                return
            if 'user_id' in df_legacy.columns and 'user_idx' in df_legacy.columns:
                df_legacy[['user_id', 'user_idx']].to_csv(user_map_csv, index=False, encoding='utf-8-sig')
                return
    except Exception:
        pass
    try:
        export_user_map_from_db(db_path, user_map_csv)
        return
    except Exception:
        pass
    try:
        pd.DataFrame(columns=['user_id', 'user_idx']).to_csv(user_map_csv, index=False, encoding='utf-8-sig')
    except Exception:
        pass


@st.cache(allow_output_mutation=True)
def load_recommender(model_path, user_map_csv):
    ensure_user_map_csv(user_map_csv, legacy_user_map_path=LEGACY_USER_MAP_PATH, db_path=DB_PATH)
    service = RecommenderService(model_path=model_path, books_path=BOOKS_PATH, user_map_path=user_map_csv, top_k=10)

    try:
        service.user_to_idx = {str(k): int(v) for k, v in getattr(service, 'user_to_idx', {}).items()}
        service.book_to_idx = {str(k): int(v) for k, v in getattr(service, 'book_to_idx', {}).items()}
    except Exception:
        pass

    try:
        model = service.model
        try:
            book_layer = model.get_layer('book_emb')
            max_book_dim = book_layer.input_dim
            current_max_idx = max(service.book_to_idx.values()) if service.book_to_idx else 0
            if current_max_idx >= max_book_dim:
                service.book_to_idx = {k: v for k, v in service.book_to_idx.items() if v < max_book_dim}
        except ValueError:
            pass
        try:
            user_layer = model.get_layer('user_emb')
            max_user_dim = user_layer.input_dim
            current_user_max = max(service.user_to_idx.values()) if service.user_to_idx else 0
            if current_user_max >= max_user_dim:
                service.user_to_idx = {k: v for k, v in service.user_to_idx.items() if v < max_user_dim}
        except ValueError:
            pass
    except Exception as e:
        print(f"❌ 自动修复模型索引时出错（非致命）：{e}")

    print(f"✅ 已加载模型：{model_path}")
    return service


def recommend_popular(top_k=10):
    results = []
    try:
        books = []
        with gzip.open(BOOKS_PATH, "rt", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if i >= 50000: break
                try:
                    b = json.loads(line)
                    books.append(b)
                except:
                    continue
        df = pd.DataFrame(books)
        if "ratings_count" in df.columns:
            df["ratings_count"] = pd.to_numeric(df["ratings_count"], errors="coerce").fillna(0)
            df = df.sort_values("ratings_count", ascending=False).head(top_k)
        else:
            df = df.head(top_k)
        for idx, r in df.iterrows():
            results.append({
                "book_id": str(r.get("book_id", f"tmp_{idx}")),
                "title": r.get("title", "未知标题"),
                "image_url": r.get("image_url", "https://via.placeholder.com/120x160.png?text=No+Cover")
            })
    except Exception:
        results = [
            {"book_id": "1", "title": "The Great Gatsby",
             "image_url": "https://images.gr-assets.com/books/1490528560m/4671.jpg"},
            {"book_id": "2", "title": "Pride and Prejudice",
             "image_url": "https://images.gr-assets.com/books/1320399351m/1885.jpg"},
        ]
    return results[:top_k]


# ------------------------- 3. Session State -------------------------
for key in ["logged_in", "user_id", "username", "recommender_loaded", "recs", "model_path"]:
    if key not in st.session_state:
        if key in ["logged_in", "recommender_loaded"]:
            st.session_state[key] = False
        else:
            st.session_state[key] = None

if st.session_state.recs is None:
    st.session_state.recs = []
if st.session_state.model_path is None:
    st.session_state.model_path = MODEL_PATH

# ------------------------- 4. 页面 UI 逻辑 -------------------------

if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("📚 图书智能推荐系统")
        st.markdown("### 请登录以获取专属推荐")

        # ⚠️ 修改点：使用 st.radio 替代 st.tabs 以兼容旧版本
        auth_choice = st.radio("选择操作", ["🔑 登录", "🆕 注册"], horizontal=True)

        if auth_choice == "🔑 登录":
            username = st.text_input("用户名", key="login_user")
            password = st.text_input("密码", type="password", key="login_pw")
            if st.button("立即登录"):
                ok, res = login_user(username, password)
                if ok:
                    st.session_state.logged_in = True
                    st.session_state.user_id = str(res)
                    st.session_state.username = username
                    st.success(f"登录成功！")
                    st.experimental_rerun()
                else:
                    st.error(res)

        else:  # 注册
            r_username = st.text_input("设置用户名", key="reg_user")
            r_password = st.text_input("设置密码", type="password", key="reg_pw")
            if st.button("注册账号"):
                ok, msg = register_user(r_username, r_password)
                if ok:
                    st.success(msg + " 请切换到“登录”进行登录。")
                else:
                    st.error(msg)

else:
    # ========================= 侧边栏 =========================
    with st.sidebar:
        st.title("⚙️ 控制台")
        st.info(f"👤 用户：**{st.session_state.username}**")
        st.markdown("---")

        if not st.session_state.recommender_loaded:
            st.warning("🔴 服务未启动")
            if st.button("🚀 启动推荐引擎", key="btn_start_engine"):
                with st.spinner("正在加载模型..."):
                    service = load_recommender(st.session_state.model_path, user_map_csv=USER_MAP_CSV)
                    st.session_state.service = service
                    st.session_state.recommender_loaded = True
                st.success("✅ 引擎已就绪！")
                st.experimental_rerun()
        else:
            st.success("🟢 服务运行中")

        st.markdown("---")
        st.subheader("生成推荐")
        search_q = st.text_input("🔍 搜索 (可选)")
        top_n = st.slider("数量", 5, 30, 10)

        btn_gen_rec = st.button("✨ 生成书单", disabled=not st.session_state.recommender_loaded)

        st.markdown("---")
        with st.expander("📊 用户画像"):
            try:
                summary = get_user_feedback_summary(st.session_state.user_id)
                if summary:
                    st.write(f"👍 {summary['likes']} | 💡 {summary['interests']} | 👎 {summary['dislikes']}")
                else:
                    st.caption("暂无数据")
            except:
                st.caption("无法获取数据")

        with st.expander("🛠️ 管理员功能"):
            if st.button("📥 导出 CSV"):
                try:
                    df = export_feedback_to_csv()
                    if df is not None: st.success(f"导出 {len(df)} 条")
                except Exception as e:
                    st.error(f"失败：{e}")

            st.markdown("---")
            if st.button("🔄 一键再训练"):
                try:
                    with st.spinner("训练中..."):
                        df = load_feedback_data()
                        df = preprocess_feedback(df)
                    prog = st.progress(0)
                    for p in range(0, 80, 10):
                        time.sleep(0.05)
                        prog.progress(p)
                    save_path, count = simple_retrain_model(df, use_csv=True)
                    prog.progress(100)
                    st.session_state.model_path = save_path
                    st.session_state.service = load_recommender(save_path, user_map_csv=USER_MAP_CSV)
                    st.success(f"完成！样本: {count}")
                except Exception as e:
                    st.error(f"失败：{e}")

        st.markdown("---")
        if st.button("🚪 退出"):
            st.session_state.logged_in = False
            st.session_state.user_id = None
            st.session_state.username = None
            st.session_state.recs = []
            st.experimental_rerun()

    # ========================= 主界面 =========================

    if btn_gen_rec and st.session_state.recommender_loaded:
        service = st.session_state.service
        uid = str(st.session_state.user_id)
        recs = []
        try:
            if hasattr(service, "user_to_idx") and uid in service.user_to_idx:
                recs = service.recommend(uid, top_k=top_n)
            else:
                # ⚠️ 修改点：使用 st.info/success 替代 st.toast 以兼容旧版本
                st.info("为您推荐热门书籍")
                recs = recommend_popular(top_k=top_n)
        except Exception as e:
            st.error(f"推荐出错，切换热门推荐: {e}")
            recs = recommend_popular(top_k=top_n)
        st.session_state.recs = recs

    st.title("📚 为您精选的书籍")
    st.markdown("---")

    if not st.session_state.recommender_loaded:
        st.info("👈 请在左侧点击 **“启动推荐引擎”**")
    elif not st.session_state.recs:
        st.info("👈 请在左侧点击 **“生成书单”**")
    else:
        recs = st.session_state.recs
        num_cols = 5

        for i in range(0, len(recs), num_cols):
            row_books = recs[i:i + num_cols]
            cols = st.columns(len(row_books))

            for col, book in zip(cols, row_books):
                title = book.get("title", "未知标题")
                img = book.get("image_url", "")
                book_id = str(book.get("book_id", f"tmp_{i}_{hash(title) % 10000}"))
                if not img: img = "https://via.placeholder.com/120x160.png?text=No+Cover"

                with col:
                    st.image(img, use_column_width=True)
                    st.markdown(
                        f"<div style='height: 45px; overflow: hidden; font-size: 13px; font-weight: bold; margin-bottom: 5px;'>{title}</div>",
                        unsafe_allow_html=True
                    )

                    # ⚠️ 修改点：按钮垂直排列 (Stack) 防止 Columns nested 报错
                    # ⚠️ 修改点：使用 st.success 替代 st.toast 以兼容旧版本
                    uid = st.session_state.user_id

                    if st.button("👍 喜欢", key=f"like_{book_id}_{i}"):
                        save_feedback(str(uid), book_id, "like")
                        st.success(f"已喜欢：{title}")

                    if st.button("💡 感兴趣", key=f"int_{book_id}_{i}"):
                        save_feedback(str(uid), book_id, "interested")
                        st.info(f"感兴趣：{title}")

                    if st.button("👎 不喜欢", key=f"dislike_{book_id}_{i}"):
                        save_feedback(str(uid), book_id, "dislike")
                        st.warning(f"不喜欢：{title}")

            st.write("")