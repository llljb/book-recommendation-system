# 个性化图书推荐系统

基于 Python、TensorFlow / Keras、Streamlit 与 SQLite 的图书推荐项目。项目负责人主要完成或修改 Web 交互、用户注册登录、反馈管理、模型调用与反馈再训练流程。

## 功能

- 注册与登录，记录用户身份与交互反馈。
- 推荐模型推理、热门图书回退、书籍信息与封面展示。
- 对新用户、映射缺失、模型索引不匹配和预测异常进行处理。
- 点赞 / 感兴趣 / 不喜欢反馈入库，反馈评分映射与 CSV 导出。
- 基于反馈再训练，保存模型并在页面重新加载。

## 结构

| 文件 | 用途 |
| --- | --- |
| `app/app.py` | Streamlit 页面与流程编排 |
| `app/recommender_service.py` | 模型加载、索引检查与推荐服务 |
| `app/user_auth.py` | SQLite 用户注册登录 |
| `app/feedback_manager.py`、`feedback_exporter.py` | 反馈存储与导出 |
| `app/retrain_manager.py` | 反馈预处理、映射与再训练 |
| `book_recommender.py`、`train_model.py` | Keras 推荐模型与训练入口 |
| `app/project_config.py` | 本地数据、模型及运行目录配置 |

## 运行

推荐 Python 3.10，使用独立虚拟环境：

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app/app.py
```

注册、登录与热门推荐回退可在缺少预训练模型时使用。个性化模型推理与再训练需要自行准备匹配的数据、模型及映射，详见 `data/README.md` 和 `saved_models/README.md`。这是源码整理版，不含训练所需的外部数据、权重或真实账户数据库；不能把缺失模型的回退展示当作个性化推荐效果。

默认数据目录为 `data/`，模型目录为 `saved_models/`，运行产生的数据库与反馈导出位于 `runtime/`。可用环境变量 `BOOK_DATA_DIR` 和 `BOOK_MODEL_DIR` 更换数据位置。

## 实现范围与整理记录

本仓库展示可核对的推荐与应用工程代码，不宣称实现标准 LightGCN 或完整 MAML，也不提供未经复现的准确率提升数字。未收录历史绘图脚本与不相关第三方工程。

本次整理统一了原工程中混用的 D/E/F 盘绝对路径，添加依赖与文档；没有改写核心推荐算法。原始数据、模型、反馈和用户数据库均保留在作者本机，不上传。认证模块采用历史 SHA-256 摘要实现，适用于本地课程展示；部署为公网服务前应升级密码存储、访问控制与运维配置。
