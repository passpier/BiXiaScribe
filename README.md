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

## 介面預覽

用瀏覽器讀、比較 `scripts/eval_generation.py` 已生成的劇本，取代肉眼開 `out/eval/*.json` 的手動流程。
共四種模式——**單篇閱讀 / 並排比較 / 總覽表 / 生成**；「生成」模式則會在背景執行緒觸發一次真正的生成，
即時顯示經過秒數、以任務為單位的進度條，還有一個真的能中斷執行的「取消」鍵。

![單篇閱讀 - 事件分頁](./docs/images/ui-single-events.webp)
*事件分頁：把觸發條件、對話台詞、分支選項渲染成可讀的散文，而不是原始 JSON。*

<table>
<tr>
<td width="50%">
<img src="./docs/images/ui-single-npc.webp" width="100%">
<em>NPC 分頁：角色表（id / 姓名 / 身分 / 性格 / 說話風格）。</em>
</td>
<td width="50%">
<img src="./docs/images/ui-single-run.webp" width="100%">
<em>執行紀錄分頁：這次生成的 <code>RunReport</code>——三個 role 各自用的模型、耗時、
<code>retrieval_calls</code>、<code>repair_attempts</code>、<code>total_tokens</code>、
<code>coerced_from</code>。</em>
</td>
</tr>
</table>

`retrieval_calls` 逐份劇本攤在 UI 上，讓上面關鍵數據段落提到的「零檢索呼叫」現象一眼就能查，
不必再翻 log。

## 功能

- RAG 索引：中文感知切塊 → embedding → Chroma，支援斷點續傳。
- Hybrid 檢索（向量 + 自寫 BM25，見下方關鍵數據）。
- 三 agent 劇本生成（編劇 → 對話 → 校對），輸出結構化 JSON + 交叉參照驗證。
- 分層/狀態化生成管線（拆書 → 排場 → 逐場寫戲，因果圖即時校驗、斷點續跑、批次確認），
  與三 agent 管線並存，設 `PIPELINE_MODE=layered` 或用 CLI/UI 的對應旗標選用。
- 模型組合 A/B 與成本估算（`scripts/eval_generation.py`），單元測試全程不打真實 API。
- 可關閉語料檢索（`RETRIEVAL_ENABLED=false` / `--no-retrieval` / UI 勾選框），省下最大宗的 token
  花費，用來 A/B 語感本身較好的模型是否真的需要語料佐證。
- Streamlit 介面，四種模式，含瀏覽器直接觸發生成——見上方[介面預覽](#介面預覽)。
- 📋 規劃中：從 UI 編輯/存回劇本、RPG Maker 匯出。

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

**檢索：hybrid vs 純向量**（`scripts/eval_retrieval.py`，14 部金庸全集 + 11 本
webnovel 索引，14 條武俠查詢）：嚴格比較（只看最相關的 1 筆，`--top-k 1`）時，兩種模式的
來源命中率一樣，但**關鍵字命中率（是否真的含確切招式/門派名）vector 只有 75%，hybrid 有
91.7%**——字元 bigram BM25 補上了向量檢索容易漏掉的專有名詞比對。（預設 `--top-k 5` 下兩者
都是 100%，這組查詢集在該粒度下太簡單看不出差異；完整結果見
[`docs/DESIGN_NOTES.md`](./docs/DESIGN_NOTES.md#檢索評估結果vector-vs-hybrid)。）

**生成：no-RAG A/B**（2026-08-19，`SCRIPT_LENGTH=long`，`--pipeline-mode legacy`，n=5/組，
`scripts/eval_generation.py --variants deepseek-v4-pro,deepseek-v4-pro-norag`）。這組兩個變體模型組合完全相同（機械角色 extractor/beat_expander 用
deepseek-v4-flash-0731，writer/dialogue-scene_writer/proof 用 deepseek-v4-pro-0423），唯一差異是
`use_retrieval`：

| | 有檢索（deepseek-v4-pro） | 無檢索（-norag） |
|---|---|---|
| 成功率 | 5/5 | 5/5 |
| 平均成本/次 | $0.0770 | $0.0545 |
| 平均 tokens | 155,450 | 47,082 |
| retrieval_calls | 10.60 | 0（依設計） |
| events / npcs | 18.2 / 6.6 | 17.0 / 4.8 |
| 對話行數 / 平均行長 | 70.0 / 34.0 字 | 51.8 / 26.1 字 |
| npc_speaking_pct | 100% | 90% |
| **usd_per_event** | **$0.0043** | **$0.0081** |
| self_loop 分支比例 | 0.0% | 20.0% |

檢索雖然讓單次成本多 4 成（多注入的語料片段佔了額外 tokens），但不只是換來「語感」：NPC 數、
台詞行數、平均行長都更高，每個 NPC 都有開口，而且 `self_loop_branch_pct`（指向自己、走不出去的
死分支）從 20% 降到 0%——這是結構性缺陷，不是文筆偏好。由於有檢索那組同時也產出更多內容，
`usd_per_event` 反而更低（$0.0043 vs $0.0081）：多花的錢在「每單位產出」上不是溢價。
（n=5、單一 rep，屬方向性訊號，self-loop 這項關聯性尤其值得用更大樣本覆核。）

一個仍然成立的非顯而易見發現：`retrieval_calls` 顯示「模型宣稱支援 function calling」不等於
「在 CrewAI 的 ReAct loop 裡真的會主動呼叫工具」——需要逐模型檢查 `retrieval_calls`，不能只看
provider 標示的能力。完整分析方法見
[`docs/DESIGN_NOTES.md`](./docs/DESIGN_NOTES.md#4-比較不同-agent-的模型組合)；逐句台詞比較
需要肉眼讀 `out/eval/` 下實際存的劇本 JSON（用下方[介面預覽](#介面預覽)的並排比較模式）。

**成本回顧**：`src/bixiascribe/pricing.py` 會對每一列精確計算 `cost_usd`（含 prompt cache 折扣，
見 `cost_basis`）。`python scripts/eval_generation.py --dry-run` 會在花費任何 token 前印出完整
矩陣的預估成本。

## 快速開始

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # 只有要跑劇本生成（OpenRouter）才需要
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
.venv/bin/streamlit run ui/app.py
```

（畫面見上方[介面預覽](#介面預覽)）

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
| Embedding | `bge-m3`（[FlagEmbedding](https://github.com/FlagOpen/FlagEmbedding)，本機、離線、免 API key） |
| 多 Agent 框架 | [CrewAI](https://www.crewai.com/) |
| LLM 路由 | [OpenRouter](https://openrouter.ai/)（透過 CrewAI 的 `LLM` + litellm `openrouter/` 前綴） |
| 資料驗證 | [pydantic](https://docs.pydantic.dev/) |

支援環境：Python ≥ 3.12（`crewai` 要求 ≥ 3.10，本 repo 統一用 3.12）。

## 授權

本專案程式碼採用 [MIT License](./LICENSE)。
