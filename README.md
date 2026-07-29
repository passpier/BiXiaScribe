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

## 為何用這套架構生成武俠劇本

跟直接丟一句 prompt 給 ChatGPT 生劇本比起來，BiXiaScribe 的差異：

- **RAG 檢索真實語料，而非純靠模型腦補武俠語感** —— 索引自建語料庫，生成對話時用檢索結果
  餵給 LLM，用詞、招式名稱更貼近原著風格。
- **中文感知的切塊器** —— 以字元數計長度、優先在段落/句讀處切分，不是照搬英文 NLP 工具的
  token 切法。
- **結構化輸出 + 自動交叉驗證** —— 劇本的 `npc_id`、`next_event_id` 等欄位互相參照用 Python
  二次檢查，不是「LLM 自己說校對過了就算過」。
- **本機優先、零成本可跑通全流程** —— 預設 embedding 是本機 `bge-m3`（離線、免費、免 API
  key）；LLM 也有 `fake` 模式，跑測試不需要真的呼叫任何模型 API。

## 關鍵數據

**Stage 1 —— hybrid 檢索 vs 純向量檢索**（`scripts/eval_retrieval.py`，14 部金庸全集 + 11 本
webnovel 索引，14 條武俠查詢）：嚴格比較（只看最相關的 1 筆，`--top-k 1`）時，兩種模式的
來源命中率一樣，但**關鍵字命中率（是否真的含確切招式/門派名）vector 只有 75%，hybrid 有
91.7%**——字元 bigram BM25 補上了向量檢索容易漏掉的專有名詞比對。（預設 `--top-k 5` 下兩者
都是 100%，這組查詢集在該粒度下太簡單看不出差異；完整結果見
[`docs/DESIGN_NOTES.md`](./docs/DESIGN_NOTES.md#檢索評估結果vector-vs-hybrid)。）

**Stage 2 —— 五組模型組合 A/B**（`scripts/eval_generation.py`，n=10/組，2026-07-29）：

| 組合 | 成功率 | retrieval_calls 平均 | 零呼叫比例 | 平均 tokens |
|---|---|---|---|---|
| baseline（三 role 皆 deepseek-chat） | 10/10 | 2.10 | 4/10 | 16,492 |
| prose-split（對話換 qwen3-235b） | 10/10 | 0.40 | 6/10 | 13,166 |
| dialogue-control-openai（對話換 gpt-4o-mini） | 10/10 | 3.30 | 0/10 | 28,002 |
| dialogue-control-qwen（對話換 qwen3-30b） | 10/10 | 0.00 | 10/10 | 10,851 |
| cheap-ends（編劇/校對換 qwen3-30b） | 0/10 | — | — | — |

預設維持 `baseline`（三個 role 都用 `deepseek/deepseek-chat`）：最快、最省、結構最豐富，
且沒有其他組合能在每個指標都贏過它。一個非顯而易見的發現：`retrieval_calls` 顯示「模型
支援 function calling」不等於「在 CrewAI 的 ReAct loop 裡真的會主動呼叫工具」——qwen 系列
模型即使官方標示支援 tool calling，實測呼叫率仍偏低甚至掛零。完整分析與逐句台詞比較見
[`docs/MILESTONES.md`](./docs/MILESTONES.md)。

## 快速開始

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # 只有要用 Gemini embedding 或 Stage 2 才需要
```

```bash
# 1. 建索引（範例語料，10 秒內跑完，免 API key）
python scripts/build_index.py --corpus tests/sample_corpus.txt

# 2. 查詢檢索結果（預設 hybrid 模式：向量 + BM25）
python scripts/test_retrieval.py --query "獨孤九劍的劍法精要" --top-k 3

# 3. 生成劇本前先零成本檢查 backend/API key/索引是否就緒
python scripts/generate_script.py --requirement "測試" --preflight-only

# 4. 生成劇本（需要 LLM_BACKEND=openrouter + OPENROUTER_API_KEY）
python scripts/generate_script.py --requirement "少林弟子下山查一樁滅門案" --out script.json

# 5. 用瀏覽器檢視/並排比較已生成的劇本（免 API key、免 token），或用「生成」模式直接觸發生成
pip install -r requirements-ui.txt
streamlit run ui/app.py
```

自己的語料放進 `data/corpus/`（不假設 UTF-8）；換語料/換 embedding backend、比較檢索與模型
組合品質的完整指令，見 [`docs/DESIGN_NOTES.md`](./docs/DESIGN_NOTES.md)。

## 輸出格式

`script.json` 結構大致如下（完整欄位定義見
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

## 技術棧

| 分類 | 技術 |
|---|---|
| 語言 | Python 3.12 |
| 向量庫 | [Chroma](https://www.trychroma.com/)（embedded `PersistentClient`，本機資料夾） |
| Embedding | `bge-m3`（[FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)，本機） / `gemini-embedding-001`（Google API） |
| 多 Agent 框架 | [CrewAI](https://www.crewai.com/) |
| LLM 路由 | [OpenRouter](https://openrouter.ai/)（透過 CrewAI 的 `LLM` + litellm `openrouter/` 前綴） |
| 資料驗證 | [pydantic](https://docs.pydantic.dev/) |

支援環境：Python ≥ 3.12（`crewai` 要求 ≥ 3.10，本 repo 統一用 3.12）。

## 專案狀態

- ✅ Stage 1：中文感知 RAG 索引（txt → 切塊 → embedding → Chroma），支援斷點續傳。
- ✅ Hybrid 檢索（向量 + 自寫 BM25，見上方關鍵數據）。
- ✅ Stage 2：三 agent 劇本生成（編劇 → 對話 → 校對），輸出結構化 JSON + 交叉參照驗證。
- ✅ 雙 backend 切換（embedding／LLM 皆有離線/免費模式），單元測試全程不打真實 API。
- ✅ Stage 3：Streamlit 介面（`streamlit run ui/app.py`）——三種唯讀模式可瀏覽、並排比較不同模型
  組合產出的劇本，取代肉眼開 JSON 的手動流程；另有「生成」模式可直接在瀏覽器輸入劇情需求、
  觸發一次真正的生成（背景執行緒跑，即時進度、可取消）。
- 📋 從 UI 編輯/存回劇本／RPG Maker 匯出（規劃中）

## 深入閱讀

- [`docs/DESIGN_NOTES.md`](./docs/DESIGN_NOTES.md) —— 設計決策理由、完整操作指令、安裝疑難排解。
- [`docs/MILESTONES.md`](./docs/MILESTONES.md) —— 進度追蹤、A/B 實驗完整數據與逐句分析。
- [`CLAUDE.md`](./CLAUDE.md) —— 給 AI coding agent 看的架構/介面說明，人類讀也一樣有用。
- [`CONTRIBUTING.md`](./CONTRIBUTING.md) —— 本地開發環境設定、測試與 lint 指令。

## 貢獻

歡迎任何形式的貢獻——bug 回報、功能建議，或直接送 code：

- 🐛 **發現 bug？** 用 [bug report 範本](https://github.com/passpier/BiXiaScribe/issues/new?template=bug_report.md)開 issue。
- 💡 **有功能建議？** 用 [feature request 範本](https://github.com/passpier/BiXiaScribe/issues/new?template=feature_request.md)，或到 [Discussions](https://github.com/passpier/BiXiaScribe/discussions) 聊聊。
- 🔧 **想貢獻程式碼？** 看 [`CONTRIBUTING.md`](./CONTRIBUTING.md)。

## 授權

本專案程式碼採用 [MIT License](./LICENSE)。

> ⚠️ 此授權僅涵蓋本 repo 的原始程式碼。`data/corpus/` 下的武俠小說語料，以及
> `bge-m3` / `gemini-embedding-001` 等第三方模型權重，並不隨本 repo 散布，
> 使用時請自行遵守其各自的授權條款。

## 聯絡

💬 [Discussions](https://github.com/passpier/BiXiaScribe/discussions) ・ 🐛 [Issues](https://github.com/passpier/BiXiaScribe/issues) ・ 👤 [@passpier](https://github.com/passpier)

> 這是一個個人 side project，目前由我獨立維護，回覆速度可能不固定，還請見諒 🙏

<div align="center">

⭐ 覺得這個專案有幫助的話，歡迎給個 star！

</div>
