<div align="center">

# BiXiaScribe

**用 RAG 檢索武俠語料，交給多 agent 生成結構化的武俠 RPG 劇本 JSON。**

[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![Stars](https://img.shields.io/github/stars/passpier/BiXiaScribe?style=social)](https://github.com/passpier/BiXiaScribe/stargazers)

[繁體中文](./README.md) | [English](./README.en.md)

</div>

---

## 這是什麼？

BiXiaScribe 是一個武俠 RPG 劇本生成器。輸入一句劇情需求（例如「少林弟子下山查一樁滅門案」），
它會從你自建的武俠小說語料庫檢索相關內容，再交給三個分工的 LLM agent（編劇 → 對話 → 校對）
產出一份結構化的劇本 JSON——含 NPC 設定、事件、分支選項、觸發條件——可作為後續遊戲製作
（如 RPG Maker）的素材來源。

跟直接丟一句 prompt 給 ChatGPT 生劇本比起來，BiXiaScribe 的差異：

- **RAG 檢索真實語料，而非純靠模型腦補武俠語感** —— 索引你自己蒐集的武俠小說文本，生成對話時
  用檢索結果餵給 LLM，用詞、招式名稱更貼近原著風格。
- **中文感知的切塊器** —— 自寫的遞迴切塊邏輯，以字元數計長度、優先在段落/句讀處切分，
  不是照搬英文 NLP 工具的 token 切法。
- **結構化輸出 + 自動交叉驗證** —— 三個 agent 產出的劇本會用 Python（而非再問一次 LLM）
  檢查 `npc_id`、`next_event_id` 等欄位互相參照是否一致，不是「看起來合理就算過」。
- **本機優先、零成本可跑通全流程** —— 預設 embedding backend 是本機的 `bge-m3`（離線、免費、
  免 API key）；LLM 也有 `fake` 模式，跑測試不需要真的呼叫任何模型 API。

### 目錄

- [安裝](#安裝)
- [使用範例](#使用範例)
- [功能](#功能)
- [技術棧](#技術棧)
- [設計筆記](#設計筆記)
- [貢獻](#貢獻)
- [授權](#授權)
- [聯絡](#聯絡)

---

## 安裝

### 前置需求

- Python ≥ 3.12（`crewai`／Stage 2 要求 ≥ 3.10，本 repo 統一用 3.12 開發）

### 安裝步驟

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

```bash
# 只有在要用 Gemini embedding 或 Stage 2（CrewAI）時才需要填 .env
cp .env.example .env
```

### 驗證安裝

不需要任何 API key、不連網路，10 秒內可確認環境裝好了：

```bash
python tests/test_chunking.py
# 預期輸出：一連串 chunking 測試案例的 PASS，最後印出 OK / 全部通過
```

<details>
<summary>常見安裝問題</summary>

- **`chromadb` 版本相關的 Rust panic**（例如 `range start index ... out of range`）：
  `crewai` 硬性要求 `chromadb~=1.1.0`，`requirements.txt` 已對齊。如果你的 `data/chroma/`
  是在更新版本的 chromadb 下建立的，打開時會 crash——刪掉 `data/chroma/` 後
  用 `python scripts/build_index.py --reset` 重建。
- **切換 `EMBED_BACKEND` 後 Chroma 報錯**：`CorpusEmbeddingFunction.name()` 把
  backend/model/維度/task_type 都編進 collection 名稱裡，不能對同一份 `data/chroma/`
  直接切換 backend——要用 `--reset` 重建索引。

</details>

---

## 使用範例

### 1. 建索引（Stage 1）

用內建的範例語料跑一次 smoke test：

```bash
python scripts/build_index.py --corpus tests/sample_corpus.txt
```

要用自己的語料，把 `.txt` 檔放進 `data/corpus/`（不假設 UTF-8，會依序嘗試
utf-8 → gb18030 → big5），然後：

```bash
python scripts/build_index.py
# 加 --reset 可清空重建整個 collection
```

索引具備斷點續傳能力：已索引過的 chunk 會被跳過、每個 batch 分次 upsert，
中斷或碰到 rate limit 後重跑是安全的。

### 2. 查詢檢索結果

```bash
python scripts/test_retrieval.py --query "獨孤九劍的劍法精要" --top-k 3
```

輸出範例：

```text
[1] distance=0.1823  source=笑傲江湖.txt
    ...獨孤九劍的精要在於「無招」，見招拆招，後發制人...

[2] distance=0.2456  source=笑傲江湖.txt
    ...風清揚傳授劍法之時，反覆強調破劍式、破刀式...
```

會印出每筆結果的距離分數、來源檔名、片段預覽，用來肉眼確認檢索結果語意相關。

### 3. 生成劇本（Stage 2）

需要已建好的索引，以及 `LLM_BACKEND=openrouter` + `OPENROUTER_API_KEY`（在 `.env` 設定）：

```bash
python scripts/generate_script.py --requirement "少林弟子下山查一樁滅門案" --out script.json
```

輸出的 `script.json` 結構大致如下（完整欄位定義見
[`src/bixiascribe/schema.py`](./src/bixiascribe/schema.py)）：

```json
{
  "title": "...",
  "premise": "...",
  "variables": [{ "id": "...", "name": "...", "initial": "..." }],
  "npcs": [{ "id": "...", "name": "...", "identity": "...", "personality": "...", "speech_style": "..." }],
  "events": [
    {
      "id": "...",
      "title": "...",
      "location": "...",
      "triggers": [...],
      "dialogue": [{ "npc_id": "...", "line": "...", "emotion": "..." }],
      "branches": [{ "id": "...", "choice_text": "...", "next_event_id": "..." }]
    }
  ]
}
```

不加 `--out` 則直接把 JSON 印到 stdout。生成完成後，`npc_id`／`next_event_id` 等交叉參照
會自動用 `schema.validate_references()` 二次檢查，不只信任 LLM 自報「校對通過」。

---

## 功能

- ✅ **Stage 1：中文感知 RAG 索引** —— txt → 中文感知遞迴切塊 → embedding（本機 `bge-m3`
  或 Gemini API）→ Chroma 向量索引，支援斷點續傳。
- ✅ **Stage 2：三 agent 劇本生成** —— 編劇（事件/分支骨架）→ 對話（RAG 檢索餵入語感）
  → 校對（schema + 交叉參照驗證），輸出結構化劇本 JSON。
- ✅ **雙 backend 切換，開發零成本** —— embedding 與 LLM 都有離線/免費模式
  （`bge-m3`、`fake` LLM），單元測試全程不打真實 API。
- 📋 **Streamlit 介面**（規劃中）

---

## 技術棧

| 分類 | 技術 |
|---|---|
| 語言 | Python 3.12 |
| 向量庫 | [Chroma](https://www.trychroma.com/)（embedded `PersistentClient`，本機資料夾） |
| Embedding | `bge-m3`（[FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)，本機） / `gemini-embedding-001`（Google API） |
| 多 Agent 框架 | [CrewAI](https://www.crewai.com/) |
| LLM 路由 | [OpenRouter](https://openrouter.ai/)（透過 CrewAI 的 `LLM` + litellm `openrouter/` 前綴） |
| 資料驗證 | [pydantic](https://docs.pydantic.dev/) |

> **為何預設 `bge-m3` 而非 Gemini？** 本機、離線、免 API key、無 rate limit，適合開發階段
> 反覆重跑索引；Gemini backend 仍保留給需要雲端 embedding 品質時使用。

> **為何透過 OpenRouter 而非各家 provider SDK？** 換模型只是改一個 env var
> （`LLM_MODEL` / `LLM_MODEL_WRITER` 等），不用改程式碼或重新串接 SDK。

支援環境：Python ≥ 3.12（`crewai` 要求 ≥ 3.10，本 repo 統一用 3.12）。

---

## 設計筆記

給同樣在學 RAG／embedding 的人：

- **向量庫選 Chroma embedded 模式**：`PersistentClient` 直接寫本機資料夾，不用另外起
  server／付費雲端服務，開發階段零成本、零維運負擔。
- **Gemini embedding 的維度與距離度量**：`gemini-embedding-001` 輸出向量會被截斷到 1536 維
  並做 L2 normalize，讓 Chroma 用 cosine 距離比較——normalize 後歐氏距離與 cosine
  距離在數學上等價，這是官方建議的標準作法，不是隨意選的。索引用 `RETRIEVAL_DOCUMENT`、
  查詢用 `RETRIEVAL_QUERY` 這兩個不同的 `task_type`，是因為 Gemini 的 embedding 模型對
  「這段文字是要被搜到的文件」vs「這段文字是搜尋請求」會用不同方式編碼，分開指定能讓
  檢索品質更好。
- **切塊器為什麼自己寫**：中文書寫沒有空白分詞，直接套用英文 NLP 工具的 token
  切法效果不好；`src/bixiascribe/chunking.py` 改用「字元數」當長度單位，並優先在
  段落／句讀處切分，純 Python 無外部依賴，方便理解與除錯。
- **索引可斷點續傳**：已索引過的 chunk ID 會被跳過，寫入採每批次 upsert，
  跑到一半斷線或碰到 API rate limit，重新執行同一個指令就能接續，不用整個重來。

---

## 貢獻

歡迎任何形式的貢獻——bug 回報、功能建議，或直接送 code：

- 🐛 **發現 bug？** 用 [bug report 範本](https://github.com/passpier/BiXiaScribe/issues/new?template=bug_report.md)開 issue。
- 💡 **有功能建議？** 用 [feature request 範本](https://github.com/passpier/BiXiaScribe/issues/new?template=feature_request.md)，或到 [Discussions](https://github.com/passpier/BiXiaScribe/discussions) 聊聊。
- 🔧 **想貢獻程式碼？** 看 [`CONTRIBUTING.md`](./CONTRIBUTING.md)，裡面有完整的本地開發環境設定、
  測試與 lint 指令。

---

## 授權

本專案程式碼採用 [MIT License](./LICENSE)。

> ⚠️ 此授權僅涵蓋本 repo 的原始程式碼。`data/corpus/` 下的武俠小說語料，以及
> `bge-m3` / `gemini-embedding-001` 等第三方模型權重，並不隨本 repo 散布，
> 使用時請自行遵守其各自的授權條款。

---

## 聯絡

- 💬 問題與討論：[GitHub Discussions](https://github.com/passpier/BiXiaScribe/discussions)
- 🐛 Bug／功能：[Issues](https://github.com/passpier/BiXiaScribe/issues)
- 👤 Maintainer: [@passpier](https://github.com/passpier)

> 這是一個個人 side project，目前由我獨立維護，回覆速度可能不固定，還請見諒 🙏

<div align="center">

⭐ 覺得這個專案有幫助的話，歡迎給個 star！

</div>
