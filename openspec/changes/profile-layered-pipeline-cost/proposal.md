## Why

A `script_length=long`、layered 模式的生成跑了 1 小時 46 分（19 場戲），引發「是不是
CrewAI 架構限制、該不該遷移到 LangChain/LangGraph」的疑慮。用真實 checkpoint 資料
（`.bixia_state/1787271639-req-30f25bc507`）與安裝的 crewai 1.15.5 原始碼實測歸因後，
發現最貴的兩項成本（beat 依賴鏈幾乎線性、每場耗時與產出大小只有 r²=0.27 的弱相關）都
不是排程器能解的問題，而現有 `RunReport`/checkpoint 完全沒有 per-scene 層級的計時或
呼叫歸因——連「847 秒到底花在哪」都答不出來。在能分辨這 73% 看不見的時間之前，任何
優化（reasoning 參數、beat DAG prompt、迴圈上限）的效果都無法驗證，遷移與否更無從
評估。現在需要把這些發現固化下來，定出可觀測性補強的範圍與優先順序。

## What Changes

- 記錄本次診斷的六項實測證據與判讀原則到 `design.md`（beat DAG 寬度、per-scene 耗時
  與產出大小的相關性、crewai 六個 agent 的預設參數盤點、OpenRouter reasoning 支援度、
  CrewAI-free 純函式模組 vs. orchestrator 的價值分布），作為「暫不遷移 LangGraph」
  這個否決決策的可查依據。
- 定義新能力 `scene-generation-observability` 的設計方向：比照 `crew/tools.py::
  RetrievalStats` / `crew/execute.py::FallbackStats` 既有的 module-level、
  thread-safe 累加器慣例，在 layered 管線的每場 scene 生成中記錄 elapsed 時間、
  reasoning token 用量、guardrail 重試次數、ReAct tool 回合數，並收進 `RunReport`
  （比照既有慣例，為讀取舊 JSONL 列提供預設值）。
- 定出後續優化的先後順序：先補齊本能力的可觀測性，再驗證 reasoning_effort 參數、
  beat DAG prompt 改寫、Agent 逾時/迴圈上限三項優化的實際效果。
- 本 change 除規劃文件外，**同時實作** `scene-generation-observability` 能力本身：
  新增 `crew/scene_metrics.py`（比照 `RetrievalStats`/`FallbackStats` 慣例的
  thread-safe per-scene 累加器）、`RunReport`/JSONL/`review.RunRecord` 新增
  `scene_metrics` 欄位、review UI 執行紀錄 tab 的對應呈現。`reasoning_effort`
  參數、beat DAG prompt 改寫、Agent 逾時/迴圈上限三項優化仍留待後續 change
  （見決策二：可觀測性是驗證它們的前提，不與它們並行）。

## Capabilities

### New Capabilities
- `scene-generation-observability`: layered 管線逐場 scene 生成的執行歸因
  （elapsed / reasoning tokens / guardrail retries / tool 回合數），累進 `RunReport`
  與 JSONL run 紀錄，讓後續的排程/prompt/模型參數優化可被量化驗證。

### Modified Capabilities
（無——本 change 不變更既有已發布的 spec 行為，只是為尚未存在 spec 的新能力定調；
`RunReport`/`review.RunRecord` 目前沒有對應的 spec 檔案。）

## Impact

- 文件層：新增 `openspec/changes/profile-layered-pipeline-cost/{proposal.md,design.md,
  specs/scene-generation-observability/spec.md,tasks.md}`。
- 程式碼層（本 change 範圍）：新增 `src/bixiascribe/crew/scene_metrics.py`；修改
  `src/bixiascribe/crew/orchestrator.py`（`_default_write_scene`/`_default_repair_scene`/
  `dispatch_next`/`dispatch_batch`/`run_layered` 掛點與 sidecar 持久化）、
  `src/bixiascribe/crew/execute.py`（`run_task` 計時/降級記錄）、
  `src/bixiascribe/crew/tools.py`（`WuxiaRetrievalTool._run` 檢索次數歸因）、
  `src/bixiascribe/crew/tasks.py`（`_scene_guardrail` 重試記錄）、
  `src/bixiascribe/crew/pipeline.py`（`RunReport.scene_metrics`）、
  `src/bixiascribe/review.py`（`RunRecord.scene_metrics`）、`ui/app.py`
  （執行紀錄 tab 新 expander）、`scripts/generate_script.py`（stderr 報告新增一行）。
- 不在本 change 範圍（留待後續 change，見決策二）：
  `src/bixiascribe/llm.py::build_llm()`（reasoning_effort 落點）、
  `src/bixiascribe/crew/agents.py`（迴圈/逾時上限落點）、
  `src/bixiascribe/crew/tasks.py::make_beat_expand_task`（beat DAG prompt 落點）、
  LangGraph 遷移（決策一：否決）。
