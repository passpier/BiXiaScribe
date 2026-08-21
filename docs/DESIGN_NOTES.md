# BiXiaScribe — 設計筆記與完整操作說明

這份文件是 [`README.md`](../README.md) 的延伸。README 只留下賣點與最短可跑路徑，所有實測數字
集中在 [`docs/BENCHMARKS.md`](./BENCHMARKS.md)；想知道「為什麼這樣設計」，或需要每個 script
的完整用法/輸出範例，看這裡。

## 設計筆記

給同樣在學 RAG／embedding 的人：

- **向量庫選 Chroma embedded 模式**：`PersistentClient` 直接寫本機資料夾，不用另外起
  server／付費雲端服務，開發階段零成本、零維運負擔。向量統一做 L2 normalize 再存進去，
  讓 Chroma 用 cosine 距離比較——normalize 後歐氏距離與 cosine 距離在數學上等價，這是
  embedding 檢索的標準作法，不是隨意選的。
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

## 通過率修復實測記錄（2026-08-21）

GMUD schema 重構（`ddc4ec9`）後，用 `deepseek-v4-flash-0731`（目前正式環境的唯一模型）跑
`flash-only` 變體，`out/generation_runs*.jsonl` 顯示真實跑全部失敗，而重構前是 12/12 成功。
以下是找出根因、驗證、修正的完整過程，供之後排查類似問題參考。

**第一步：確認根因不在 schema.py 的必填欄位**——`Event` pydantic 層只有 4 個必填欄位、
`Script` 只有 2 個，但實際送給 provider 的 JSON schema 不是這個。CrewAI 的
`generate_model_description()` 對 `output_pydantic` 呼叫 `ensure_all_properties_required()`，
把每個欄位都標成必填（`strict: true`），無視 schema.py 裡的 `default=""`。實測
`Event`：wire schema 14/14 必填（Branch 11/11、SkillCheck 8/8）。這代表模型每次都要吐出所有
「其實通常是空字串」的欄位，速度變慢；而 provider 端的 strict enforcement 不保證真的執行，
一旦漏一個欄位就在 openai SDK 的 `parse_chat_completion` 直接拋 `ValidationError`，發生在
`task.execute_sync()` **內部**，繞過了既有的三層救援機制（`_coerce_model`）。

**第二步：兩次真實跑驗證，均使用 `flash-only`（六 role 全 deepseek-v4-flash-0731）、
layered、`script_length=short`：**

1. 第一次跑在 guardrail 層就死掉——`check_scene_rpg` 的 `known_npc_ids` 只從
   `session.character_cards` 取，從沒把 `session.player_card` 併進去，導致玩家一講話
   （`npc_id="player"`）就被誤判成未登場的 NPC，3 次 retry 全部燒光。已修正
   （`crew/tasks.py`）。
2. 修正後重跑，直接打到目標 `ValidationError`：`exc.errors(include_url=False)` 完整印出
   兩個 branch，內容完整且前後呼應（`cost`/`immediate_feedback`/`payoff_description`/
   `converges_to_event_id: "ev-ch1-converge"` 全部填好），唯獨沒有 `next_event_id`
   這個 key。同時佐證「慢」的抱怨：`script_length=short`、只有 2 個 branch，
   `elapsed=2082s`、`reasoning_tokens=39062`。

**第三步：三個修正**：
1. `src/bixiascribe/wire.py`——寬鬆鏡像類別當 `output_pydantic`，每個欄位都有預設值，
   讓 SDK 的 parse 永遠不因缺欄位而硬失敗；缺欄位變成可檢視的空字串，而不是例外。
2. `src/bixiascribe/crew/normalize.py`——在 `validate_references()` 之前跑機械式修復：
   `next_event_id` 依序嘗試從 `converges_to_event_id` → 所屬 chapter 的
   `converge_event_id` → 事件序列中下一個 event 回填；`Script.chapters` 為空但多個
   event 引用同一個 chapter_id 時反向補建骨架；純標註性 dangling id 直接清空。
   拿診斷跑到的真實資料（branch 有 `converges_to_event_id` 但無 `next_event_id`）
   重建測試案例，確認正是 fallback 鏈第一順位命中。
3. `SessionDocument.allowed_ids`——把合法 id 當封閉選單餵給 scene_writer prompt，
   從源頭減少「引用不存在的 id」這類問題。

**判讀原則記錄**：`crew/pipeline.py::_coerce_model`/`crew/tasks.py::_coerce_for_guardrail`
既有的三層救援（pydantic → json_dict → raw_scan）對「provider 端 strict enforcement 沒生效
導致 `ValidationError` 直接在 `execute_sync()` 內部拋出」這個失敗模式完全沒用——例外發生在
它們能接手之前。這是本次診斷最重要的一課：**Task 拋出的例外，要先確認是在
`execute_sync()` 內部炸的還是外部救援層漏接的，兩種問題的修法完全不同。**

## Phase 2：對照《武俠單人劇本生成範例》做 schema 瘦身（2026-08-21）

Phase 1 把 `deepseek-v4-flash-0731`（六 role 全同一模型）、layered、`script_length=long` 這組
正式環境配置從 0/3 失敗救到乾淨過關，但那趟跑了 ~2 小時、1.5M tokens、76 次請求——診斷時已經
知道原因：`ensure_all_properties_required()` 讓每個欄位都變成 wire-required，模型每個物件都要
吐出一堆「其實通常是空字串」的欄位。對照 `docs/武俠單人劇本生成範例.md`（單人武俠 GMUD
框架），這份指南完全沒有 regions/sub-locations、沒有任務系統、每個分支也沒有獨立的
condition 欄位——這些正是目前 schema 裡最重、卻對這份指南沒有對應概念的部分。

**刪除的欄位（不是 deprecate，是直接刪）**：`Region`/`SubLocation`/`Quest` 三個 class 整個
移除；`Branch.condition`/`.payoff_chapter_id`、`SkillCheck.kind`/`.difficulty`/
`.item_bypass_id`、`Chapter.beat_ids`/`.clue_ids`、`NPC.attitude_by_threshold`、`Clue.serves`、
`ExtractionResult.props`/`.branch_candidates`。實測 wire schema 縮減：`Script` 16.8KB → 13.6KB
（−19%）、`Event` 4.9KB → 4.0KB（−18%）、`ExtractionResult` 10.8KB → 8.5KB（−21%）；必填欄位數
`Event` 14→11、`Branch` 11→9、`SkillCheck` 8→5。

**唯一沒有照原計畫刪的欄位：`Branch.effects`**。原本規劃裡打算跟 `.condition` 一起刪，但先查了
`crew/causal.py::event_to_node()`——它把 `branches[*].effects` 當成 `PlotNode.postconditions`
的主要來源，`effect_ops` 只會渲染成 `"target_id：op=value"` 這種機械化事實字串（例如
`'心境值：add=-15'`），永遠不會跟觸發條件衍生的 precondition 產生衝突比對。如果連 `effects`
都刪掉，因果一致性檢查（`check_scene_consistency`）會直接失效——這是這次瘦身唯一不能犧牲的
品質底線，所以 `effects` 保留。

**唯一數值**：指南要求「只有一個心境值/正邪值這類數值，切成幾個區間」，但這次沒有動
`PlayerCharacter.stats: list[Variable]` 的 schema 本身（避免影響 full-profile 情境下的彈性），
改用 prompt 措辭（`crew/tasks.py::_GMUD_WORLD_CLAUSE` 改寫成要求「恰好一個 stat + 恰好 3 條
不重疊的 stat_threshold」）加上新的離線 guardrail `guardrails.check_single_stat()`（併入
`collect_quality_problems()`，report-only，不進 in-loop 重試）。

**`_SCHEMA_VERSION` 2 → 3**：跟上次 GMUD 框架加欄位時的做法一樣——正在跑的舊 checkpoint 直接
判定為「沒有 checkpoint」重新開始，不嘗試遷移。已產出的 `out/eval/*.json`／`.bixia_state/`
劇本檔案不受影響：`schema.py` 沒設 `model_config`，pydantic 預設的 `extra="ignore"` 讓舊檔案
裡多出來的 `regions`/`quests` 等欄位讀取時直接被忽略，不會炸掉。

**尚未驗證的部分**：以上都是離線可驗證的結構性改動（377→378 個離線測試全過、`ruff` 乾淨、
legacy/layered 兩條管線在 `LLM_BACKEND=fake` 下都能跑出 `validate_references()==[]` 的完整
劇本）。schema 變小是否真的讓真實模型呼叫變快、guardrail/repair 重試次數變少，還沒有花真錢
驗證過——下一步是拿 Phase 1 那組同樣的 production 配置（`flash-only`、layered、
`script_length=long`）重新跑一次，比較 elapsed/tokens/repair_attempts，才能確認間接效益
（更少必填欄位 → 更少幻覺 id → 更少 repair pass）是否真的兌現。

## Phase 3：從 UI 生成觸發的截斷 JSON 修復（2026-08-21）

從 Streamlit 的「生成」模式跑一次 layered 生成，終端機印出兩段
`OpenAI API call failed: 1 validation error for LenientExtractionResult / Invalid JSON: EOF while
parsing an object`，`input_value='{\n '`——不是漏欄位，是 provider 回應本身被截斷成 3 個字元。
跟通過率修復實測記錄那次不同：那次是模型漏填一個欄位（JSON 本身完整），這次是 JSON 根本沒收尾，
`wire.py` 的寬鬆鏡像對這種情況無效（沒東西可解析）。crewai 自己的 `Agent.max_retry_limit`
（預設 2）會重跑同一種呼叫形狀，這次记录里前兩次都截斷、第三次才成功，代表這個問題不是每次都會
發生，但每次重試都是一次完整的高價呼叫，三次都截斷就會整趟失敗。

**修法**：`crew/execute.py::run_task()` 在 `Agent.max_retry_limit` 降到 1（每個 task 兩次
structured 嘗試機會）後多包一層——遇到結構化輸出解析失敗，同一個 task 改用不帶
`output_pydantic` 的自由文字版本（schema 說明改寫進 `expected_output`）重試一次，讓
`_coerce_model` 既有的 raw_scan 救援真的有機會派上用場。六個 `make_*_task()` 都加了
`structured: bool = True` 參數，`structured=True` 的 prompt 文字逐字不變。`STRUCTURED_OUTPUT`
（`.env`，預設 `auto`）是手動退路：`off` 讓六個 task 一開始就走自由文字。

**離線驗證這一步意外挖出一個更深的既有 bug**：用 `STRUCTURED_OUTPUT=off` + `LLM_BACKEND=fake`
跑 layered 管線驗證自由文字路徑，`_default_extract` 回傳 `npcs=[]`，但 FakeLLM 的罐頭資料明明
有兩個 NPC。追下去發現 `schema.parse_model_json()`「保留掃到的最後一筆合法比對」這個規則，對
`ExtractionResult` 這種每個欄位都有預設值的 schema 是壞的——因為 `extra="ignore"`，任何一個
dict 都能驗證通過，包括 `npcs` 陣列裡的某一個 NPC 物件本身（在原始文字裡的位置比外層物件晚）。
結果「最後一筆」變成一個被誤判的巢狀片段，蓋掉了真正完整的頂層答案。這個 bug 原本就存在（結構化
輸出失敗時本來就會落到 raw_scan），只是很少真的被踩到——正常情況下 `output.pydantic` 就有值，
根本不會掃到這一層；這次修復把自由文字模式變成六種 task 的常態路徑之一，才讓它第一次在離線驗證
就現形。修法：改成保留「span 最大」的合法比對（同分時仍偏好較晚出現的那筆，維持原本「跳過模型
前置說明文字」的行為）——父物件的 span 必然嚴格包含、因此不會小於它自己任何一個巢狀片段的
span，所以真正的頂層答案永遠會贏過巢狀片段。

**驗證**：`STRUCTURED_OUTPUT=off` + `LLM_BACKEND=fake` 下 legacy 與 layered 兩條管線都能跑出
`validate_references()==[]` 的完整劇本；390 個離線測試（含新增的 12 個）全過，`ruff` 乾淨。真錢
驗證（同一句劇情需求重跑 UI 生成，確認不再出現未接住的 `Invalid JSON`）留給實際使用時機決定。

## Phase 4：layered 管線 per-scene 執行歸因（2026-08-21）

一趟 `script_length=long`、layered 模式的生成跑了 1 小時 46 分（19 場戲），引發「是不是
CrewAI 架構限制、該不該遷移到 LangChain/LangGraph」的疑慮。用真實 checkpoint 資料離線分析
歸因後（見 `openspec/changes/profile-layered-pipeline-cost/design.md` 完整六項證據），發現：

- **beat 依賴鏈幾乎線性**：19 個 beat 拆成 17 個 batch，有效並行度僅 ≈1.1x，遠低於
  `SCENE_CONCURRENCY=3` 允許的上限——`plan_batches()`/`dispatch_batch()` 的排程層在這趟
  真實資料上幾乎沒有用武之地。
- **場次耗時與產出大小只有弱相關（r²=0.27）**：用 checkpoint 檔案的 mtime 差反推每場耗時，
  對照輸出 JSON 字元數，`r(輸出大小, 耗時) = 0.519`。最慢的一場（847 秒）產出量反而低於
  中位數，代表 ~73% 的耗時花在生成內容以外的地方（reasoning token、guardrail 重試、
  ReAct tool 回合、結構化輸出降級重試四者之一或全部），且**當時完全沒有任何欄位能分辨是
  哪一項**——連「847 秒到底花在哪」都答不出來。
- 五個懷疑症狀中四個是「CrewAI 已有一級參數（`max_iter`/`max_execution_time`/
  `max_rpm`/`reasoning_effort`）但未被設定」，不是框架限制；排程層的理論收益在真實資料上
  只值 ~1.7%。**結論：否決遷移 LangGraph**，改為在 CrewAI 既有 API 內逐項調參。

在能分辨這 73% 看不見的時間之前，任何後續優化（`reasoning_effort` 參數、beat DAG prompt
改寫、Agent 逾時/迴圈上限）的效果都無法驗證——會退化成肉眼盯著總 elapsed 猜測，而非
`docs/BENCHMARKS.md` 既有方法論要求的「每次只變動一個變因、用結構化指標歸因」。

**修法**：`src/bixiascribe/crew/scene_metrics.py`，比照 `crew/tools.py::RetrievalStats`/
`crew/execute.py::FallbackStats` 既有的 module-level、`threading.Lock` 累加器慣例，記錄
每場戲的 `elapsed_s`/`call_elapsed_s`/`repair_elapsed_s`/`llm_calls`/`reasoning_tokens`/
`guardrail_retries`/`retrieval_calls`/`structured_fallbacks`，鍵為 beat id，收進
`RunReport.scene_metrics`（JSONL/`review.RunRecord` 比照既有慣例補上舊列預設值），在
review UI 執行紀錄 tab 呈現為依耗時排序的表格，並在 `generate_script.py` 的 stderr 報告
加上「最慢 3 場」摘要。細節與設計取捨見 CLAUDE.md「Per-scene 執行歸因」一節。

**驗證**：`LLM_BACKEND=fake` 下離線跑完整 layered run（`scene_metrics` 數量與
`scenes_generated` 相符、`.bixia_state/<run_id>/scene_meta_*.json` sidecar 逐場產生）、
resume 同一 run_id（沿用先前 process 寫下的 sidecar，`scene_metrics` 仍完整）、
`SCENE_CONCURRENCY=1` 序列路徑、legacy 模式（`scene_metrics == []`）四種情境皆通過；
412 個離線測試（含新增的 22 個，覆蓋 thread-local scope 的併發正確性、resume 語意、
guardrail/structured-fallback per-scene 歸因）全過，`ruff` 乾淨。真實模型的實測數字
（拿 `deepseek-v4-*` 對照這份儀表）是下一個 change 的範圍，本次只交付可觀測性本身。

## 為何是這些技術選擇

> **為何用本機 `bge-m3`？** 本機、離線、免 API key、無 rate limit，適合開發階段
> 反覆重跑索引，且免費可無限次重建。

> **為何透過 OpenRouter 而非各家 provider SDK？** 換模型只是改一個 env var
> （`LLM_MODEL` / `LLM_MODEL_WRITER` 等），不用改程式碼或重新串接 SDK。

## 完整操作說明

### 1. 建索引

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

### 3. 生成劇本

需要已建好的索引，以及 `LLM_BACKEND=openrouter` + `OPENROUTER_API_KEY`（在 `.env` 設定）。
下真正的單前，可以先用 `--preflight-only` 零成本確認 backend／API key／索引都就緒：

```bash
python scripts/generate_script.py --requirement "測試" --preflight-only
python scripts/generate_script.py --requirement "少林弟子下山查一樁滅門案" --out script.json
```

生成完成後會在 stderr 印出一份執行報告（各 agent 使用的模型、耗時、token 用量、校對修復次數、
`wuxia_corpus_search` 被呼叫的次數）——`retrieval_calls` 為 0 就代表對話 agent 這次沒有實際
用到語料庫檢索，通常是 `LLM_MODEL_DIALOGUE` 不支援 function calling，或雖支援但在 CrewAI 的
ReAct loop 裡沒有實際被選用（見 [`docs/BENCHMARKS.md`](./BENCHMARKS.md) 的模型組合 A/B 數據）。

不加 `--out` 則直接把 JSON 印到 stdout。生成完成後，`npc_id`／`next_event_id` 等交叉參照
會自動用 `schema.validate_references()` 二次檢查，不只信任 LLM 自報「校對通過」；若發現問題，
校對 agent 會拿到具體錯誤再修一次（最多兩次），修不好才會回報失敗，而不是整趟生成直接作廢。

#### RPG 遊戲性：玩家/屬性/道具/任務

早期產出讀起來像小說大綱，不像可以「玩」的 RPG 腳本，根因是 `schema.py` 原本沒有玩家、
數值屬性、道具、任務的一級位置——模型只能自己想辦法繞：捏造一個 `npc_player`/`npc_narrator`
假 NPC 頂替玩家/旁白、`variables` 全是布林旗標、extractor 抽出的 `props` 從未真正進入最終
劇本、NPC 全部在第一個事件就同時開口講話。現在 `schema.py` 有了 `PlayerCharacter`
（`stats: list[Variable]`，`kind="stat"` 的數值屬性，如內力/聲望/銀兩）、`Item`
（`acquired_in_event_id`）、`Quest`（`event_ids`）、`EffectOp`（結構化的分支效果，取代原本
純文字的 `Branch.effects`），以及 `NPC.first_appearance_event_id`/`introduction`。
`Script.player`/`.items`/`.quests` 是最終輸出的位置；layered 管線的 `ExtractionResult` 帶著
同一組欄位，由 `orchestrator.py::_assemble_script()` 直接複製進最終 `Script`。

光靠 prompt 要求並不夠——CrewAI 沒有「skill」機制，但有 `Task(guardrail=..., 
guardrail_max_retries=N)`：一個純 Python callable，檢查沒通過就回傳 `(False, 中文修正指示)`，
讓 CrewAI 當場帶著回饋重試該 task，比事後才跑的校對修復迴圈更早介入。
`src/bixiascribe/crew/guardrails.py` 是這組檢查（`check_script_rpg`／`check_extraction_rpg`／
`check_scene_rpg`），純函式、零 crewai import，掛在 `crew/tasks.py` 的
`make_writer_task`／`make_extract_task`／`make_scene_write_task` 上，由 `GUARDRAILS_ENABLED`
（`.env`，預設 `true`）／`GUARDRAIL_MAX_RETRIES`（預設 `2`）控制。**`LLM_BACKEND=fake` 時一律
關閉**（不管 `GUARDRAILS_ENABLED` 設什麼）——`FakeLLM` 的罐頭回應永遠不可能滿足 RPG 檢查，
掛著只會讓每次離線測試白白重試到 `GUARDRAIL_MAX_RETRIES` 次。

### 4. 比較不同 agent 的模型組合

三個 agent（編劇／對話／校對）可各自指定不同模型（`LLM_MODEL_WRITER`／`_DIALOGUE`／`_PROOF`），
但一次只改一個 env var、重跑一次程序很難做系統性比較。`scripts/eval_generation.py` 從
`eval/model_variants.json` 讀取多組模型組合，逐一對 `eval/script_requirements.txt` 裡的劇情需求
生成劇本，把每次執行的 token 用量、`retrieval_calls`、結構性指標（事件/NPC/台詞數、NPC 開口比例、
GMUD 框架涵蓋率——`branches_with_cost_pct`／`branches_with_payoff_pct`／`checks_with_fallback_pct`／
`main_scene_ratio`／`events_with_clue_pct`／`chapters_with_convergence_pct`／
`stat_threshold_coverage_pct`／`faction_count`／`ending_count`，見 `crew/metrics.py::gmud_metrics()`）
都記錄成一行 JSON，累積寫進 `out/generation_runs.jsonl`，並印出各組合的彙總比較表：

```bash
# 先零成本檢查每組模型 id、API key、索引都就緒
python scripts/eval_generation.py --dry-run
# 真的跑一組矩陣（範例：目前進行中的 no-RAG A/B）
python scripts/eval_generation.py --variants deepseek-v4-pro,deepseek-v4-pro-norag --repeat 1
# 只想重新看彙總表，不想再花錢
python scripts/eval_generation.py --from-jsonl out/generation_runs.jsonl
```

這些都是結構性指標，不是 LLM-as-judge 的文字品質評分——實際台詞是否夠「武俠」，仍需要肉眼讀過
`out/eval/` 下存的劇本 JSON，見下一節的檢視 UI。詳見 `CLAUDE.md`「Comparing per-agent model splits」
一節；目前跑過的完整結果見 [`docs/BENCHMARKS.md`](./BENCHMARKS.md)。

**若要跨 provider 比較**：`LLM_PROVIDER_ONLY` 是行程層級的 env var（`config.py` import 時讀取一
次），單一個 `eval_generation.py` invocation 沒辦法讓矩陣裡每組 variant 各自 pin 不同 provider——
要跑這種橫跨多 provider 的矩陣，得對每組分別下指令（`LLM_PROVIDER_ONLY=<provider>
python scripts/eval_generation.py --variants <name>`），都指定同一個 `--jsonl` 路徑就會全部
append 到同一份紀錄。目前 `eval/model_variants.json` 裡的變體都不 pin provider（走 OpenRouter
預設路由），這一段只在之後又新增 pin 特定 provider 的變體時才用得到。

### 5. 檢視/比較已生成的劇本，以及從 UI 觸發生成

上一節產出的 `out/eval/*.json` 用肉眼一份份開 JSON 讀太慢，`ui/app.py` 是 Streamlit 介面
（`out/` 為 gitignored，clone 下來需要自己先跑過 `eval_generation.py` 或 UI 的生成模式才有資料
可讀）：

```bash
pip install -r requirements-ui.txt   # streamlit 獨立放這個檔，不進核心 requirements.txt
.venv/bin/streamlit run ui/app.py
```

四種模式：單篇閱讀（事件/NPC/變數/玩家道具任務/章節/勢力/地圖/線索/真相/結局/門檻表/執行紀錄/
原始 JSON 分頁，`validate_references()` 結果直接顯示在最上面）、並排比較（同一個劇情需求下，多個
模型組合的劇本左右對照）、總覽表（所有紀錄的結構性指標一次看完，含上面的 GMUD 框架涵蓋率）——這
三種**唯讀**，不呼叫 pipeline、不需要 API key、不載入 Chroma；以及生成（輸入劇情
需求、選模型變體，直接在瀏覽器裡跑一次真正的生成）——這個模式跟 CLI 一樣，需要 API key 與 Chroma
索引，會花費 token。

資料層 `src/bixiascribe/review.py`（唯讀瀏覽）與觸發生成的 `src/bixiascribe/generation.py` 都刻意不 import streamlit——武俠 RPG 劇本 RAG 架構方案文件裡，Streamlit 只是這個階段的「臨時駕駛艙」，核心邏輯不該被綁死在特定前端上。`out/eval/*.json` 的檔案會被之後的 rep覆寫，所以 `out/generation_runs*.jsonl` 裡記錄的 `script_metrics()` 數字可能已經過期——UI 一律用磁碟上目前的檔案重新計算，不直接信任 JSONL 裡的數字。

生成模式跑在背景執行緒（`generation.GenerationJob`），因為一次真正的生成要 126–240 秒，而且
CrewAI 的 `step_callback` 對這個 crew 完全不會觸發（編劇/校對兩個 agent 沒有工具、走的是
`_invoke_loop_native_no_tools`，這條路徑直接跳過 `_invoke_step_callback`；已對照安裝的 crewai
1.15.5 原始碼驗證）——同步阻塞的做法在這 2–4 分鐘裡只能重繪 3 次（`task_callback`，每個任務結束觸發
一次），畫不出會跳動的計時器。背景執行緒讓 `ui/app.py` 能用 `st.fragment(run_every=1.0)` 每秒輪詢
`job.snapshot()`，顯示即時計時、任務進度條，以及真正有效的「取消」按鈕；生成出來的劇本與執行紀錄
在背景執行緒裡就直接寫檔（`out/eval/ui-{variant}__{slug}.json` +
`out/generation_runs_ui.jsonl`——後者是獨立檔案但仍符合 `RUN_LOG_GLOB` 的 glob，會被既有的檢視模式
自動抓到，同時不會混進 eval 工具的 A/B 統計），所以就算瀏覽器中途重新整理弄丟了 UI 的追蹤狀態，
剛才花掉的 token 產出的劇本也不會不見，重新整理後在單篇閱讀模式仍找得到。

分層管線的批次確認畫面不只列 beat id——`generation.GenerationJob.pending_scenes()`/`.scene_context()`
（薄包裝，底層是 `crew/orchestrator.py` 兩個純讀檔函式 `load_pending_scenes()`/`load_scene_context()`）
把本批 staged 場次的完整內容（標題、地點、觸發條件、台詞、分支）讀出來，UI 直接沿用既有的
`_render_event()` 渲染，確認前看得到寫了什麼，不是盲按。這個面板也刻意畫在
`_render_generation_progress()` 的 `@st.fragment(run_every=1.0)` 之外——fragment 每秒自動重繪一次，
按鈕點擊有機率被下一次重繪蓋掉；`_render_generation_progress()` 一偵測到
`snap.awaiting_confirmation` 就用整頁的 `st.rerun()`（不是 `scope="fragment"`）離開 fragment，交給
頁面主體的靜態分支渲染確認面板。

不檢索語料庫（`RETRIEVAL_ENABLED`，見 CLAUDE.md「Indexing and retrieval」一節）想量化的問題是：
「換成語感本身較好的模型後，語料檢索還值不值得付出的 token 成本？」量測方法：

```bash
python scripts/eval_generation.py --variants baseline,baseline-norag --repeat 3
# 或用目前的生產規模檔位（SCRIPT_LENGTH=long）：
python scripts/eval_generation.py --variants deepseek-v4-pro,deepseek-v4-pro-norag --repeat 1
```

每一對變體的其他角色模型都完全相同，唯一差異是 `use_retrieval`，所以彙總表的 token/成本差異可以
直接歸因於檢索本身；肉眼讀 `out/eval/*.json` 的台詞差異則回答「省下的成本是否用犧牲語感換來
的」。跑完後把 avg tokens / avg cost / 肉眼判斷補進 [`docs/BENCHMARKS.md`](./BENCHMARKS.md)。
`baseline-norag`/`deepseek-v4-pro-norag` 對 UI 的模型變體選單都隱藏
（`ui_visible: false`），只透過 `--variants <name>-norag` 執行。

## 安裝疑難排解

- **`chromadb` 版本相關的 Rust panic**（例如 `range start index ... out of range`）：
  `crewai` 硬性要求 `chromadb~=1.1.0`，`requirements.txt` 已對齊。如果你的 `data/chroma/`
  是在更新版本的 chromadb 下建立的，打開時會 crash——刪掉 `data/chroma/` 後
  用 `python scripts/build_index.py --reset` 重建。
- **改了 `LOCAL_EMBED_MODEL` 後 Chroma 報錯**：`CorpusEmbeddingFunction.name()` 把
  backend/model/維度/task_type 都編進 collection 名稱裡，`data/chroma/` 的 collection
  跟目前的 embedding 設定是綁定的——換了 embedding model 要用 `--reset` 重建索引，
  不能直接對同一份索引沿用舊資料。

## 檢索評估結果（vector vs hybrid）

完整表格與解讀已搬到 [`docs/BENCHMARKS.md`](./BENCHMARKS.md#1-檢索hybrid-vs-純向量)；跑法：
`python scripts/eval_retrieval.py`（加 `--top-k 1` 做嚴格比較）。
