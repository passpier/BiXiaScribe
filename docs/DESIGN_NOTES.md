# BiXiaScribe — 設計筆記與完整操作說明

這份文件是 [`README.md`](../README.md) 的延伸。README 只留下賣點、關鍵數據與最短可跑
路徑；想知道「為什麼這樣設計」，或需要每個 script 的完整用法/輸出範例，看這裡。

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
- **為什麼自己刻 BM25 而不是裝 `rank_bm25` + `jieba`**：中文分詞函式庫（如 jieba）沒有
  自訂詞典時，容易把「獨孤九劍」這種專有名詞切成「獨孤／九劍」甚至更破碎，反而失去加關鍵字
  檢索的意義。改用「中文字元 bigram」（獨孤九劍 → 獨孤／孤九／九劍）當 token，查詢字串用
  同樣方式切，天然就能完整比對到專有名詞，不需要維護詞典。融合向量與 BM25 兩種分數時用
  **Reciprocal Rank Fusion**（只看排名、不看原始分數）而非直接加權平均，因為 cosine 距離
  跟 BM25 分數的數值尺度完全不可比——RRF 剛好迴避了這個問題。

## 為何是這些技術選擇

> **為何預設 `bge-m3` 而非 Gemini？** 本機、離線、免 API key、無 rate limit，適合開發階段
> 反覆重跑索引；Gemini backend 仍保留給需要雲端 embedding 品質時使用。

> **為何透過 OpenRouter 而非各家 provider SDK？** 換模型只是改一個 env var
> （`LLM_MODEL` / `LLM_MODEL_WRITER` 等），不用改程式碼或重新串接 SDK。

## 完整操作說明

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

會印出每筆結果的距離分數、來源檔名、片段預覽，用來肉眼確認檢索結果語意相關。預設走
`hybrid` 模式（向量檢索 + BM25 關鍵字檢索用 Reciprocal Rank Fusion 融合），加 `--mode vector`
可比較純向量模式的結果。

想跨多個查詢比較兩種模式的品質，而不是一次只肉眼看一筆：

```bash
python scripts/eval_retrieval.py
# --top-k 1 可看差異更明顯的嚴格比較，見下方「檢索評估結果」
```

會跑 `eval/retrieval_eval.jsonl` 裡預先準備好的武俠查詢集，印出兩種模式的
source-hit@k／term-hit@k／MRR 對照表。

### 3. 生成劇本（Stage 2）

需要已建好的索引，以及 `LLM_BACKEND=openrouter` + `OPENROUTER_API_KEY`（在 `.env` 設定）。
下真正的單前，可以先用 `--preflight-only` 零成本確認 backend／API key／索引都就緒：

```bash
python scripts/generate_script.py --requirement "測試" --preflight-only
python scripts/generate_script.py --requirement "少林弟子下山查一樁滅門案" --out script.json
```

生成完成後會在 stderr 印出一份執行報告（各 agent 使用的模型、耗時、token 用量、校對修復次數、
`wuxia_corpus_search` 被呼叫的次數）——`retrieval_calls` 為 0 就代表對話 agent 這次沒有實際
用到語料庫檢索，通常是 `LLM_MODEL_DIALOGUE` 不支援 function calling，或雖支援但在 CrewAI 的
ReAct loop 裡沒有實際被選用（見下方「檢索評估結果」旁的 Stage 2 A/B 數據）。

不加 `--out` 則直接把 JSON 印到 stdout。生成完成後，`npc_id`／`next_event_id` 等交叉參照
會自動用 `schema.validate_references()` 二次檢查，不只信任 LLM 自報「校對通過」；若發現問題，
校對 agent 會拿到具體錯誤再修一次（最多兩次），修不好才會回報失敗，而不是整趟生成直接作廢。

### 4. 比較不同 agent 的模型組合

三個 agent（編劇／對話／校對）可各自指定不同模型（`LLM_MODEL_WRITER`／`_DIALOGUE`／`_PROOF`），
但一次只改一個 env var、重跑一次程序很難做系統性比較。`scripts/eval_generation.py` 從
`eval/model_variants.json` 讀取多組模型組合，逐一對 `eval/script_requirements.txt` 裡的劇情需求
生成劇本，把每次執行的 token 用量、`retrieval_calls`、結構性指標（事件/NPC/台詞數、NPC 開口比例
等，見 `crew/metrics.py`）都記錄成一行 JSON，累積寫進 `out/generation_runs.jsonl`，並印出各組合的
彙總比較表：

```bash
# 先零成本檢查每組模型 id、API key、索引都就緒
python scripts/eval_generation.py --dry-run
# 真的跑一組矩陣（範例只挑兩組模型比較）
python scripts/eval_generation.py --variants baseline,prose-split --repeat 1
# 只想重新看彙總表，不想再花錢
python scripts/eval_generation.py --from-jsonl out/generation_runs.jsonl
```

這些都是結構性指標，不是 LLM-as-judge 的文字品質評分——實際台詞是否夠「武俠」，仍需要肉眼讀過
`out/eval/` 下存的劇本 JSON，見下一節的檢視 UI。詳見 `CLAUDE.md`「Comparing per-agent model splits」
一節，完整的 Phase A/B/C 分析與逐句台詞比較見 [`docs/MILESTONES.md`](./MILESTONES.md)。

### 5. 檢視/比較已生成的劇本（Stage 3）

上一節產出的 40+ 份 `out/eval/*.json` 用肉眼一份份開 JSON 讀太慢，`ui/app.py` 是唯讀的 Streamlit
檢視器：

```bash
pip install -r requirements-ui.txt   # streamlit 獨立放這個檔，不進核心 requirements.txt
streamlit run ui/app.py
```

三種模式：單篇閱讀（事件/NPC/變數/執行紀錄/原始 JSON 分頁，`validate_references()` 結果直接顯示在
最上面）、並排比較（同一個劇情需求下，多個模型組合的劇本左右對照）、總覽表（所有紀錄的結構性指標
一次看完）。全程不呼叫 pipeline、不需要 API key、不載入 Chroma。

資料層 `src/bixiascribe/review.py` 刻意不 import streamlit——武俠 RPG 劇本 RAG 架構方案文件裡，
Streamlit 只是這個階段的「臨時駕駛艙」，之後要換 Tauri 桌面版，核心邏輯不該被綁死在特定前端上。
另外，`out/eval/*.json` 的檔案會被之後的 rep 覆寫，所以 `out/generation_runs*.jsonl` 裡記錄的
`script_metrics()` 數字可能已經過期——UI 一律用磁碟上目前的檔案重新計算，不直接信任 JSONL 裡的數字。

## 安裝疑難排解

- **`chromadb` 版本相關的 Rust panic**（例如 `range start index ... out of range`）：
  `crewai` 硬性要求 `chromadb~=1.1.0`，`requirements.txt` 已對齊。如果你的 `data/chroma/`
  是在更新版本的 chromadb 下建立的，打開時會 crash——刪掉 `data/chroma/` 後
  用 `python scripts/build_index.py --reset` 重建。
- **切換 `EMBED_BACKEND` 後 Chroma 報錯**：`CorpusEmbeddingFunction.name()` 把
  backend/model/維度/task_type 都編進 collection 名稱裡，不能對同一份 `data/chroma/`
  直接切換 backend——要用 `--reset` 重建索引。

## 檢索評估結果（vector vs hybrid）

實跑 `python scripts/eval_retrieval.py`（2026-07-29，本機 `bge-m3`，索引為 14 部金庸全集 +
11 本 capped webnovel，`eval/retrieval_eval.jsonl` 的 14 條查詢、12 條有 ground truth）：

**`--top-k 5`（預設）：兩種模式打平，觸底效應**

| 模式 | source-hit@5 | term-hit@5 | MRR |
|---|---|---|---|
| vector | 100.0% | 100.0% | 1.000 |
| hybrid | 100.0% | 100.0% | 1.000 |

12 條 ground-truth 查詢在 top-5 兩種模式都全中——這組查詢集在 k=5 下太容易，看不出差異
（觸底效應），不是「hybrid 沒有用」的證據。

**`--top-k 1`（嚴格比較）：hybrid 在專有名詞比對上明顯領先**

| 模式 | source-hit@1 | term-hit@1 | MRR |
|---|---|---|---|
| vector | 100.0% | 75.0% | 1.000 |
| hybrid | 100.0% | 91.7% | 1.000 |

把 top-k 收緊到 1（只看最相關的一筆），兩種模式的來源命中率一樣，但**關鍵字命中率
（term-hit，即最相關那個 chunk 的文字裡是否真的包含查詢的武俠專有名詞）vector 只有
75%，hybrid 有 91.7%**——這正是 BM25 字元 bigram 融合設計要解決的問題：向量檢索有時
會撈到語意相關但沒提到確切招式/門派名稱的段落，BM25 的關鍵字比對把含有確切詞彙的
chunk 排到更前面。

跑法：`python scripts/eval_retrieval.py --top-k 1`。
