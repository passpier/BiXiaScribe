# 貢獻指南

歡迎任何形式的貢獻——回報 bug、提建議，或直接送 PR。

## 🐛 發現 bug？

用 [bug report 範本](https://github.com/passpier/BiXiaScribe/issues/new?template=bug_report.md)
開一個 issue，附上重現步驟、環境（`EMBED_BACKEND` / `LLM_BACKEND`、Python 版本）與完整錯誤訊息。

## 💡 有功能建議？

用 [feature request 範本](https://github.com/passpier/BiXiaScribe/issues/new?template=feature_request.md)，
或先到 [Discussions](https://github.com/passpier/BiXiaScribe/discussions) 聊聊想法。

## 🔧 想貢獻程式碼？

### 本地開發環境

```bash
# 1. Fork 並 clone
git clone https://github.com/<your-username>/BiXiaScribe.git
cd BiXiaScribe

# 2. 建立 venv 並安裝依賴（含 dev 工具）
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
pip install -r requirements-ui.txt   # 僅開發 Stage 3 UI（ui/app.py）時需要

# 3. 建立分支
git checkout -b feature/your-feature-name
```

### 跑測試

```bash
# 純 chunking 單元測試，免 API key、免網路
python tests/test_chunking.py

# 全部測試（含 CrewAI pipeline，用 LLM_BACKEND=fake 跑，一樣免 key/免網路/免費）
pytest tests/
```

### Lint

```bash
ruff check .
```

送 PR 前請確保 `ruff check .` 與 `pytest tests/` 都能通過。

### 程式慣例

- 本 repo 沒有 `pyproject.toml`/`setup.py`，套件不會被 `pip install`。任何要 `import bixiascribe`
  的新 script 或 test，開頭都要加：
  ```python
  import sys
  from pathlib import Path
  sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
  ```
- 中文語料檔案編碼不保證是 UTF-8（常見 gb18030 / big5），處理文字讀取時請比照
  `indexer._read_text_any_encoding` 的容錯順序。
- Stage 3 UI 的邏輯放 `src/bixiascribe/review.py`（純 Python，不得 import streamlit），
  `ui/app.py` 只放 widget——之後要換前端（見 CLAUDE.md）時才不用重寫資料層。
  `tests/test_review.py` 同理不得 import streamlit。
- 更多背景與各 Stage 的設計決策，見 [`CLAUDE.md`](./CLAUDE.md)。

### 4. 送出 PR

Commit message 請說明「為什麼」而不只是「做了什麼」。PR 描述請包含你怎麼測試的（例如跑了哪些指令、
是否驗證過索引/生成結果）。

---

再次感謝你願意花時間讓 BiXiaScribe 變得更好 🙏
