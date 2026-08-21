## Context

見 `proposal.md` - Why。本節只記錄支撐設計決策的實測證據，全部來自對真實 checkpoint
`.bixia_state/1787271639-req-30f25bc507`（`少林弟子下山查一樁滅門案`、
`script_length=long`、`--pipeline-mode layered`、19 場戲）的離線分析，以及對已安裝
crewai 1.15.5 原始碼、OpenRouter `/api/v1/models` 的實測查驗。每個數字都附上可重跑的
指令，供之後覆核。

### 證據一：`SCENE_CONCURRENCY=3` 對這趟 run 實際失效

`plan_batches()`（`src/bixiascribe/crew/orchestrator.py:756`）用 `Beat.causal_deps`
做 Kahn-style level ordering。這趟 run 的 `beats.json` 顯示 19 個 beat 幾乎是一條線：

```
bt-order → bt-departure → bt-arrive → bt-search ┬→ bt-salvage ─┐
                                                  └→ bt-autopsy ─┴→ bt-yamen → ... → bt-track ┬→ bt-heal ──────┐
                                                                                                └→ bt-ling-trust ┴→ bt-confront → ... → bt-fate
```

19 個 beat → 17 個 batch，其中 15 個 batch 只有 1 個 beat，最大寬度是 2（只出現
2 次：`{bt-salvage, bt-autopsy}`、`{bt-heal, bt-ling-trust}`）。有效並行度
≈ 19/17 ≈ 1.1x，遠低於 `SCENE_CONCURRENCY=3` 允許的上限。

重算指令：
```bash
python3 -c "
import json
d = json.load(open('.bixia_state/<run_id>/beats.json'))['data']
for b in d['beats']: print(b['id'], b.get('causal_deps'))
"
```

### 證據二：耗時與產出大小只有弱相關（r²=0.27）

用 scene checkpoint 的 mtime 差當每場實際耗時（目前沒有任何 per-scene 計時被記錄，
這正是這個 change 要補的缺口），對照該場輸出 JSON 的字元數：

| beat | 秒 | JSON 字元 | 秒/字元 |
|---|---|---|---|
| bt-ashes | 45 | 1,997 | 0.022（最快） |
| bt-salvage | 136 | 3,536 | 0.039 |
| bt-showdown | 744 | 6,149（最大） | 0.121 |
| bt-departure | 564 | 2,824 | 0.200 |
| **bt-fate** | **847（最慢）** | **3,501（低於中位數 3,798）** | **0.242（最高）** |

`r(輸出大小, 耗時) = 0.519`，`r² = 0.27`——輸出大小只解釋了 27% 的耗時變異。
秒/字元離散度達 11 倍（0.022–0.242）。最慢的一場產出量反而低於中位數，代表
~73% 的時間花在生成內容以外的地方（reasoning token、guardrail 重試、ReAct
tool 回合、結構化輸出降級重試四者之一或全部），且現有紀錄無法分辨是哪一項。

場次階段總耗時：08:54:48 → 10:40:10（106 分鐘），平均 335 秒/場。

### 證據三：五個懷疑點中，四個 CrewAI 已有一級參數但未被設定

實測已安裝的 crewai 1.15.5：

```python
Agent.model_fields["max_iter"].default              # 25   (ReAct 迴圈上限，未覆寫)
Agent.model_fields["max_execution_time"].default     # None (無 wall-clock 上限，未覆寫)
Agent.model_fields["max_rpm"].default                 # None (未覆寫)
Agent.model_fields["max_retry_limit"].default          # 2   (crew/agents.py 六個工廠已覆寫為 1)
LLM.model_fields 含: reasoning_effort / thinking / timeout / max_completion_tokens
                                                        # 均未在 llm.py::build_llm() 使用
```

`src/bixiascribe/crew/agents.py` 六個 `make_*_agent()` 只覆寫了 `max_retry_limit=1`，
其餘四項留在框架預設值。這代表「retry 無上限」「ReAct 迴圈失控」兩個症狀的根因是
**未設定既有參數**，不是 CrewAI 缺少這個能力。

### 證據四：reasoning 是可關的請求參數，不需換模型

OpenRouter `/api/v1/models` 回應中：

```
deepseek/deepseek-v4-flash-0731.supported_parameters 含
  ["reasoning", "reasoning_effort", "include_reasoning", ...]
deepseek/deepseek-v4-pro-0813.supported_parameters 含同樣三項
```

兩顆正式環境用的模型都支援關閉/調整 reasoning，不是強制 reasoning 的模型。
`llm.py::build_llm()` 已有把 provider-routing 塞進 `additional_params.extra_body`
的既有路徑（`LLM_PROVIDER_ONLY`/`LLM_PROVIDER_SORT` 用的那條），`reasoning_effort`
可以走 `crewai.LLM` 的原生欄位，不需要新的傳遞機制。

### 證據五：retry/repair 的疊乘結構

單一 scene 生成目前疊了四層各自獨立的重試/修復：

```
execute.run_task()                             ← crew/execute.py，本專案自寫
└─ structured 嘗試
   └─ Task.execute_sync()
      └─ Agent.max_retry_limit = 1              → 最多 2 次呼叫
         └─ guardrail_max_retries = 2           → 最多 3 輪（每輪含上面 2 次）
            └─ ReAct max_iter = 25（未覆寫）      → 每次呼叫內最多 25 個 tool 回合
└─ 失敗 → free-text 重建再跑一次整組             → 再一組同樣的預算
└─ _validate_scene() 因果修復（causal.py）× 2    ← 又兩個完整 task
```

每一層都有獨立存在的正當理由（見 `CLAUDE.md`「Structured-output parse failures」
一節），不是要拆掉；但目前沒有任何欄位記錄「這場戲實際觸發了哪幾層」，只能看到
最終 elapsed。這正是可觀測性缺口的具體樣貌。

### 證據六：架構價值分布——為何遷移成本與收益不成比例

```
crew/orchestrator.py      1,477 行   checkpointer + DAG 排程 + 批次確認閘門（框架相依）
crew/guardrails.py          461 行   純函式，零 crewai import
crew/causal.py               420 行   純函式，零 crewai import
crew/context_builder.py      305 行   純函式（SessionDocument 組裝）
crew/normalize.py            120 行   純函式，零 crewai import
wire.py                      210 行   純函式，零 crewai/config import
crew/agents.py                195 行   六個 Agent 工廠（框架相依，但薄）
crew/tasks.py                 470 行   六個 Task 工廠 + prompt 組裝（框架相依）
crew/execute.py               155 行   結構化輸出降級重試（框架相依但薄，見上）
```

真正承載「因果一致性」「GMUD 品質檢查」「wire schema 寬鬆化」這些領域邏輯的模組
全部與 CrewAI 解耦，遷移框架不影響它們。CrewAI 實際只負責「組 prompt、呼叫 LLM、
解析結構化輸出」三件事（`agents.py`+`tasks.py`+`execute.py` ≈ 820 行）。

遷移到 LangGraph 的主要成本是重寫 `orchestrator.py` 的 1,477 行——它的
checkpointer 語意（`detect_stage()` 永遠從磁碟重新推導，而非增量存取）與批次確認
閘門（`stage_pending`/`confirm_batch()`/`reject_batch()`）都是這個專案特有的行為，
LangGraph 的 checkpointer 不會原生提供對等物。而收益（消除 `dispatch_batch()` 的
level barrier）在證據一那條近乎線性的圖上，實測只值
`bt-salvage` 等 `bt-autopsy` 的 108 秒／106 分鐘 ≈ **1.7%**。

### 證據七：一趟真實取消掉的 UI 生成，暴露預估缺口與一個檢索歸因漏洞

`scene-generation-observability` 實作完成後第一次真正派上用場：`out/generation_runs_ui.jsonl`
最後一列（`ts=1787310447`，`variant=ui-flash-only`）對應 checkpoint
`.bixia_state/1787309292-req-d232acf2d8`（`layered`／`script_length=long`／
`deepseek-v4-flash-0731`，六個 role 同一顆模型）：使用者在第 19.2 分鐘取消，只花了
$0.0039（37,667 tokens，4 次 LLM 呼叫），`completed_scene_ids` 是空的（第一場戲仍是
staged pending）。

用該 run 自己已完成的兩個階段（extract $0.00077／beat_expand $0.00107）加上唯一一場
scene 的實測（$0.00209、227 秒）外推：`beats.json` 已有 30 個 beat，完成全部大約
**$0.065、114 分鐘**（加上前兩階段約 2 小時）。用 `plan_batches()`
（`orchestrator.py:808`）對這 30 個 beat 重算分層，結果是 **30 個批次、每批寬度都是
1**——與證據一那趟 19/17≈1.1x 更糟，`SCENE_CONCURRENCY=3` 對這次完全沒有效果。
**結論：錢從來不是使用者按下取消的原因（$0.07 不貴），兩小時看不到終點才是。**

核對這場戲的 `scene_meta_ev-ch1-beat-01.json` sidecar 時，另外發現一個真實的歸因
漏洞：run 層的 `RunReport.retrieval_calls` 記到 3（3 條具體中文檢索 query，內容明顯
屬於這場戲），但該場的 sidecar `retrieval_calls` 卻是 **0**。追查 root cause：
`crew/scene_metrics.py` 原本用 `threading.local()` 記錄「目前在生成哪個 beat」，但
crewai 自己的原生工具呼叫迴圈（`crewai/agents/crew_agent_executor.py` 約第 746 行）
會用 `ThreadPoolExecutor.submit(contextvars.copy_context().run, ...)` 把同一輪
LLM 回應裡的多個工具呼叫併發送到執行緒池——`copy_context().run()` 只會把呼叫端的
`contextvars.ContextVar` 值帶進新執行緒，`threading.local()` 狀態不會跟著過去（用
一段最小重現腳本實測驗證：在送進 `ThreadPoolExecutor` 之前設好的
`threading.local()` 屬性，在 pool worker 裡讀回 `None`；同樣操作換成
`contextvars.ContextVar` 則能正確讀回原值）。`WuxiaRetrievalTool._run()`
剛好就是在這個 pool worker 上執行，所以它永遠看不到 `_default_write_scene()` 在
ReAct 迴圈自己的執行緒上用 `scene_scope()` 設好的 beat id。修法：把
`scene_metrics.py` 的 `_current` 從 `threading.local()` 換成
`contextvars.ContextVar`——這不影響 `dispatch_batch()` 本身跨 worker 執行緒的
場次隔離（每個 worker 執行緒的 context 本來就是各自獨立起始的），但補上了
crewai 內部這一層併發工具呼叫的傳遞路徑。

n=2（這場戲是這個模組第一次記到的兩筆非 fake 真實資料之一）：兩場都是各自 run 的
第一場、都被取消，證據二那組 r²=0.27 的迴歸還無法用這兩筆重新驗證，需要一趟真正跑
完的長 run。但兩筆一致顯示 `call_elapsed_s / elapsed_s ≈ 99.99%`
（guardrail 重試、因果修復、結構化降級三者皆為 0），且 completion token 中
reasoning token 占比約 71%——時間幾乎全部落在「那一次成功的 LLM 呼叫本身」，
與決策二排定的優先序（可觀測性之後先驗證 `reasoning_effort`）方向一致。

## Goals / Non-Goals

**Goals:**
- 為「暫不遷移 LangGraph」這個決策留下可查核的實測依據（見上六項證據），避免日後
  重新討論時要重跑一次同樣的診斷。
- 定義 `scene-generation-observability` 能力的設計方向，讓後續 change 能直接進
  tasks.md 實作，不需要重新做技術選型。
- 定出四項候選優化（可觀測性 / reasoning 參數 / beat DAG prompt / 迴圈上限）的
  驗證順序，並說明為何可觀測性是前置條件而非並行項。
- 定義 `run-cost-estimation` 能力：把 `scene-generation-observability` 補上的
  逐場歸因資料，接到生成前/生成中的使用者可見預估（見證據七），而不只是躺在
  `RunReport`/JSONL 裡供事後分析。

**Non-Goals:**
- 不在本 change 實作任何程式碼變更（`RunReport`/`orchestrator.py`/`llm.py`/
  `tasks.py` 的實際修改留給後續 change 的 tasks.md）。
- 不重新評估 LangChain（僅 LangGraph 被使用者提及，且結論同樣適用：問題不在
  agent 框架本身）。
- 不涵蓋 legacy 三 agent 管線——它沒有 `SCENE_CONCURRENCY`/`plan_batches()` 這類
  排程層，本次診斷的證據一、二、六都是 layered 管線特有的。
- 不重新設計 `guardrail_max_retries`/`CAUSAL_VALIDATION` 的既有重試策略本身，只
  補上「記錄這些重試發生了幾次」的觀測層。

## Decisions

**決策一：不遷移 LangGraph（否決）。**
理由：五個懷疑症狀中四個是「CrewAI 已有參數但未設定」（證據三），不是框架限制；
排程層的理論收益在真實資料上只有 ~1.7%（證據六）；價值密集的邏輯已經是
crewai-free 純函式（證據六），遷移不會讓這些模組變好，只會讓 1,477 行框架相依的
`orchestrator.py` 需要重寫一次且要重新驗證 checkpoint/確認閘門的既有行為。
替代方案（維持 CrewAI + 逐項調參）在 CrewAI 既有 API 內即可完成，成本遠低於遷移。

**決策二：可觀測性優先於三項優化，作為驗證它們的前提，而非與它們並行。**
理由：證據二顯示產出大小只解釋 27% 的耗時變異，代表 73% 的耗時來源目前無法歸因。
在無法歸因之前，reasoning_effort/beat DAG prompt/迴圈上限三項優化即使做了，也
無法回答「這次變快是哪個機制的功勞」——會退化成肉眼盯著總 elapsed 猜測，而不是
可覆核的 A/B（跟 `docs/BENCHMARKS.md` 既有的 A/B 方法論一致：每次只變動一個變因，
用結構化指標而非肉眼判讀來歸因）。

**決策三：per-scene 累加器沿用 `RetrievalStats`/`FallbackStats` 既有慣例，而非
新開一套機制。**
理由：`crew/tools.py::RetrievalStats` 與 `crew/execute.py::FallbackStats` 已經
是本專案「module-level、threading.Lock 保護、run 開始時 reset、run 結束時讀回
`RunReport`」的慣例，且已被 `RunReport`/JSONL/`review.RunRecord` 三層消費過一次
（`retrieval_calls`/`structured_fallbacks` 就是先例）。新的 per-scene 觀測層
（暫定 `crew/scene_metrics.py`，欄位含 `scene_elapsed_sec`/`reasoning_tokens`/
`guardrail_retries`/`tool_rounds`，鍵為 beat id）比照同一套慣例，是後續 change
tasks.md 階段的實作範圍，本 change 只鎖定這個方向。
考慮過的替代方案：直接把計時塞進 `orchestrator.py` 的 `dispatch_batch()`/
`_default_write_scene()` 區域變數，不獨立成模組——否決，因為
`dispatch_batch()` 用 `ThreadPoolExecutor` 平行跑多個 scene，區域變數無法跨執行緒
彙總，且會破壞 `orchestrator.py`「純排程，不管統計」的現有分工（`tools.py`/
`execute.py` 都是外部模組被動累加，`orchestrator.py` 只在 run 開始/結束呼叫
reset/get）。

**決策四：`reasoning_effort` 透過 `crewai.LLM` 原生欄位傳遞，而非 `extra_body`。**
理由：證據四顯示 `LLM.model_fields` 已有 `reasoning_effort`/`thinking` 欄位，
是一級參數，不需要走 `LLM_PROVIDER_ONLY` 那條「未經真實回應驗證」的
`extra_body` 路徑（`llm.py::build_llm()` 註解已明說那條路徑的不確定性）。
這降低了驗證成本：一級欄位若不生效會是明確的 crewai/litellm 層錯誤，而
`extra_body` 若無效是靜默的行為劣化。

**決策五：預估以「本次實測」優先、「歷史紀錄」次之、「固定先驗」最後，且任何一層
都不得回傳捏造的 `$0`/`0 秒`。**
理由：呼應 `pricing.estimate_cost()` 既有的 `basis` 字串慣例（"by_role"/"uniform"/
"unknown_price"，見 `pricing.py`），`estimate.py` 延伸出同一優先序給「尚未花費」的
預估：`measured_run`（這趟 run 自己已完成場次的實測平均，`scene-generation-
observability` 的直接消費者）→ `history_mode_length`（`out/generation_runs*.jsonl`
同模式同篇幅）→ `history_mode`（同模式、依 `length.LengthSpec.events_scale` 縮放
跨篇幅）→ `prior`（無任何歷史時的固定值，來源就是證據七這兩筆真實 sidecar 資料，
docstring 註明 n=2，不是憑空編造）→ `unknown_price`（有 token 數但無定價，回報
`None` 而非 0）。`scripts/eval_generation.py` 舊有的 `_estimate_matrix_cost()`
已經有一個「歷史優先、無歷史退回固定值」的預估，但它的 `_BASE_TOKENS` 常數是寫死的
舊數字，docstring 卻宣稱「scaled from 歷史平均」——與程式碼實際行為不符，且完全不
估時間；這次直接淘汰該常數，改為所有呼叫端（UI 表單、UI 進度列、`eval_generation.py
--dry-run`、`generate_script.py` 的 preflight/pre-run 印出）共用 `estimate.py` 這
一份實作。

**決策六：layered 管線的批次寬度（parallelism）由呼叫端用既有的
`orchestrator.py::plan_batches()` 提供，`estimate.py` 本身不重新實作因果排程。**
理由：`plan_batches()` 已經是這個排程邏輯唯一、被 `dispatch_batch()` 實際使用、且
已有測試覆蓋的實作（見證據一、證據六的價值分布——`crew/orchestrator.py` 是排程層
價值最集中的地方）。若 `estimate.py`（刻意設計成不依賴 crewai/checkpoint I/O 的純
函式模組，方便被 UI/CLI 兩邊共用）自己重寫一份 Kahn 分層，等於在兩個地方各自定義
「兩個 beat 算不算能並行」，一旦其中一份漏改就會產生自相矛盾的預估。因此
`estimate.estimate_run()`/`estimate_remaining()` 只接受呼叫端算好的
`batch_widths: list[int]`（每個批次幾個 beat）；沒有真實 `beats.json` 可讀時
（例如生成前的表單預估），退化為「每個 beat 各自一批」，也就是並行度 1.0——這不是
隨便選的保守假設，是證據一、證據七兩趟真實 run 都觀察到的常見情況。

## Risks / Trade-offs

- **[風險] per-scene 觀測層本身增加每場的記憶體/序列化開銷** → 緩解：欄位是純量
  （int/float/list[str]），比照 `RetrievalStats.queries` 的量級，對 19 場戲的
  run 而言可忽略；不記錄完整 prompt/response 內容。
- **[風險] `ThreadPoolExecutor` 並行寫入累加器可能有 race** → 緩解：延用
  `RetrievalStats`/`FallbackStats` 已驗證過的 `threading.Lock` 模式，不需要新的
  併發原語。
- **[風險] `reasoning_effort` 調低可能犧牲 GMUD 結構品質**
  （`branches_with_cost_pct`/`checks_with_fallback_pct` 等指標下降）
  → 緩解：後續 change 應比照 `docs/BENCHMARKS.md` 現有方法論，用
  `eval_generation.py` 跑 A/B（reasoning_effort 開/關兩組），同時看
  `crew/metrics.py::gmud_metrics()` 的結構指標，不能只看 elapsed 變快就採用。
- **[風險] beat DAG prompt 改寫若強推「更寬的圖」，可能製造出模型自己想像的假
  平行性（beat 之間其實有隱含依賴但沒宣告），導致因果不一致** → 緩解：這正是
  `check_scene_consistency()`/`CAUSAL_VALIDATION` 已經在做的事，beat DAG prompt
  的改寫必須讓 `check_beat_expand_rpg` guardrail 或新增等價檢查同步收緊，不能只
  改 prompt 措辭；這個平衡點留給實際改寫 beat_expand prompt 的後續 change 處理。
- **[取捨] 本 change 不修程式碼，意味著「先裝儀表」要再等一個 change 週期才看得到
  實際數字** → 接受：這是決策二的直接後果，換取的是後續三項優化都能被結構化驗證，
  而不是像本次診斷一樣事後用 mtime 反推。

## Migration Plan

不適用——本 change 純規劃，無執行環境變更。後續實作 change 的 rollout 應遵循
`CLAUDE.md` 既有慣例：新欄位一律加預設值、JSONL 舊列補預設值（`RunReport`/
`review.RunRecord` 已有此慣例，如 `structured_fallbacks`/`guardrails_enabled`
在既有代碼中的先例），不需要 schema version bump（純觀測欄位，不影響
`validate_references()`/checkpoint 的必填結構）。

## Open Questions

- `scene_elapsed_sec` 該包含 guardrail 重試/因果修復耗時在內的「這場戲的總掛鐘
  時間」，還是只算最後成功那一次呼叫？兩者都有用途（前者回答「這場戲花了多久」，
  後者回答「模型單次呼叫要多久」），暫定兩者都記，具體欄位命名留給實作階段的
  tasks.md 決定，不影響本 change 的範圍或方向。
- `tool_rounds`（ReAct 回合數）在 crewai 1.15.5 是否有現成的可讀取途徑（例如
  `TaskOutput`/callback 暴露的欄位），還是要靠 `WuxiaRetrievalTool._run()` 的
  呼叫次數間接估計——需要在實作階段動手驗證，屬於技術細節，不影響設計方向。
