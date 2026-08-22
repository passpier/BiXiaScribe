## Context

見 `proposal.md` — Why。本節記錄支撐設計決策的實測數字，以及與既有模組邊界相關的取捨。

### 實測一：磁碟上劇本檔案的可讀率

```
out/eval/*.json (12 份)         ValidationError × 12
.bixia_state/*/script.json (8 份) ValidationError × 8
------------------------------------------
20/20 全數無法透過 Script.model_validate() 讀取
```

原因：`2026-08-22-slim-script-schema-mvp` 是 breaking rename（`meta` 全新必填欄位、
`DialogueLine.npc` 取代必填的 `npc_id`），該 change 自己的 design.md 決策四已明確接受「舊資料讀取
失敗、不轉檔」。本 change 不重新討論那個決定——這裡的功能是「讓使用者能對這 20 份損壞檔案做點什麼
（刪除/取代)」，而不是修好讀取。

### 實測二：checkpoint 的 schema 版本分布

```
_SCHEMA_VERSION（crew/orchestrator.py:93）現行 = 4

1786717492-req-0728cc739f   v1  stage=done
1787294412-req-30f25bc507   v3  stage=done
1787294510-req-0728cc739f   v3  stage=done
1787294786-req-ccd026d6dd   v3  stage=scenes   <- 看起來「可續跑」
1787305223-req-0728cc739f   v3  stage=done
1787305511-req-0728cc739f   v3  stage=done
1787305512-req-0728cc739f   v3  stage=done
1787305581-req-0728cc739f   v3  stage=done
1787305635-req-0728cc739f   v3  stage=done
1787305976-req-4e61dbe167   v3  stage=scenes   <- 看起來「可續跑」
1787309292-req-d232acf2d8   v3  stage=scenes   <- 看起來「可續跑」
```

三個 `stage=scenes` 的目錄若被天真地當作可續跑，`load_checkpoint()`（`orchestrator.py:130-151`）
會因版本不符回傳 `None`，`detect_stage()` 因此回報 `"extract"`——**續跑會靜默地從拆書重新開始，
花費與全新執行相同的 token**，而使用者以為自己只是接續一個已完成 30 分之 1 的執行。這是本 change
唯一一個「不硬擋就會直接燒真錢」的路徑。

### 實測三：crewai 的 `reasoning_effort` 是一級欄位

```python
from crewai import LLM
LLM.model_fields["reasoning_effort"]
# annotation: Optional[Literal['none', 'low', 'medium', 'high']], default: None
```

`crewai/llm.py:794` 把它併入送給 provider 的 completion 參數，`llm.py:798` 用
`{k: v for k, v in params.items() if v is not None}` 濾掉 `None`——因此 `reasoning_effort=None`
與今日完全不傳這個參數逐位元組相同。litellm 的 OpenRouter 路徑（`providers/openai/completion.py:745-746`）
把它轉成 `params["reasoning"] = {"effort": ...}`。這比原本設計草案考慮的 `extra_body` 途徑更安全：
不需要動 `llm.py:140` 那行整包指派 `additional_params` 的程式碼，不存在與既有 provider-routing
`extra_body` 合併衝突的風險，也不用碰 repo 自己在 `llm.py:126-133` 記載的「extra_body 未經真實
OpenRouter 回應驗證」的既有風險註記。

### 實測四：`generation._cost_models()` 已經知道 layered 用 4 個 role

`generation.py:326-340` 的 layered 分支已回傳 `{"extractor", "beat_expander", "scene_writer", "proof"}`
——因為 `crew/orchestrator.py::_default_repair_scene()`（:528）在 `CAUSAL_VALIDATION != "off"`
（預設值）時，每一場戲都會呼叫校對 agent 做因果修復。`ui/app.py::_render_run_meta()` 的 layered
分支（:183-193）只顯示 3 個，是顯示層落後於既有邏輯的 bug，不是本 change 新引入的行為。

## Goals / Non-Goals

**Goals:**
- 讓使用者能刪除/匯出/匯入劇本，不需要碰檔案系統或改程式碼裡的硬編碼路徑。
- 修正單篇閱讀顯示的 role 數與 `generation._cost_models()` 的既有邏輯一致（消除重複定義）。
- 加入一個全域 reasoning-effort 旋鈕，預設完全不改變現行行為，寫進執行紀錄以便日後比較。
- 生成頁的模型輸入限制在已測試、已定價的模型範圍內。
- 續跑功能對 schema 版本不符的檢查點**拒絕**而非只警告。
- 自訂篇幅四個欄位的說明文字據實反映各自實際生效的管線模式。

**Non-Goals:**
- 不轉檔/修復既有的 20 份損壞劇本檔案——那是 `slim-script-schema-mvp` 已決議接受的行為。
- 不做 per-role 的 reasoning effort（使用者已確認整次執行一個全域設定即可）。
- 不把 variant/篇幅/檢索設定持久化進 `.bixia_state/<run_id>/state.json`——見決策四。
- 不改動 `pricing.py` 的計價邏輯——見決策六。
- 不做「編輯後存回」——README 的規劃中項目仍保留這一項與 RPG Maker 匯出。

## Decisions

**決策一：`ModelChoice` 加 `reasoning_effort` 欄位，不新增獨立參數。**
`ModelChoice` 已被串到每一個 `build_llm(role, models)` 呼叫點；新增獨立參數要改
`agents.py`/`tasks.py`/`pipeline.py`/`orchestrator.py` 約 9 個簽章，而它們都不檢視這個值。代價是
`ModelChoice` 的語意略為擴張（從「用哪個模型」到「用哪個模型、想多用力」），接受。

**決策二：`REASONING_EFFORT` 預設 `"default"`（不送參數），不是 `"none"`。**
若預設成 `"none"`，repo 內每一次真實 LLM 呼叫都會靜默改變行為（明確關閉 provider 的預設推理），
且會汙染既有的 Phase-4 A/B 基準（`flash-only` 變體、`docs/BENCHMARKS.md` 的既有數字）。`"default"`
讓這個 change 對現行行為是逐位元組不變的 no-op，直到使用者主動選擇別的檔位。

**決策三：mutation（刪除/匯入）放新模組 `library.py`，不加進 `review.py`。**
`review.py` 模組 docstring 開宗明義是唯讀索引，約 35 個既有測試建立在「這個模組任何地方都能安全
呼叫、不會有副作用」的前提上。把 `unlink()` 放進去會破壞這個不變式，也會讓
`tests/test_review.py` 的 no-streamlit 斷言之外多一條隱性的「no-mutation」規則需要另外守護。
`library.py` 依賴 `review.py`（讀取/命名）與 `generation.py`（命名慣例/rep 分配），維持單向依賴。

**決策四：續跑不把 variant/script_length/use_retrieval/reasoning_effort 持久化進
`PipelineState`/`state.json`。**
要做到這件事，需要改動 `run_layered()` 內每一個 `save_checkpoint()` 呼叫點，讓它們額外攜帶這些
執行期設定，範圍與本 change 其餘部分不成比例。改為 UI 明確警告「此檢查點沒有記錄原本的設定，續跑
會套用你現在選的設定，可能與已完成階段不一致」。`requirement` 例外：它本來就存在於
`PipelineState.requirement`，UI 直接讀出並鎖定不可編輯——因為 `run_layered()` 會把**傳入的**
requirement 餵給尚未生成的場次，改動它會讓劇本前後兩段回答不同的問題。

**決策五：`deepseek-chat` 在下拉選單中保留可選（標記為 baseline/V3 對照組），`glm-5.2` 不可選。**
`eval/model_variants.json` 的 `baseline`/`baseline-norag` 兩個變體仍在使用 `deepseek-chat`；若把它
從選單移除但留在 variants 檔裡，catalog 就不再是「單一真實來源」——變體檔能跑的模型，UI 卻選不到。
`glm-5.2` 已確認不支援結構化 JSON schema 輸出（`z-ai/glm-5.2` 會把 JSON 包進 ```json fence，卡死
`output_pydantic`），標記 `status="unusable"` 並從 `selectable()` 排除，但 `describe()` 仍可查
（歷史執行紀錄若曾用過它，渲染不會退化）。

**決策六：不修改 `pricing.py` 的計價邏輯。**
`pricing._usage_tokens()`（`pricing.py:149-164`）只讀 `prompt_tokens`/`completion_tokens`/
`cached_prompt_tokens`。OpenRouter 把 reasoning token **計在 `completion_tokens` 內**，
`ModelPrice.cost()` 已經按 completion 費率把它算進去了；若另立一個 reasoning 計價項目會造成
**重複計費**。這個 change 缺的是能見度（reasoning token 占比看不到），不是計價方式錯誤——用
單篇閱讀新增的 caption 補上能見度即可，明文記錄於此以免日後有人「修好」一個沒壞的東西。

## Risks / Trade-offs

- **[風險，已緩解] 續跑對 schema 版本不符的檢查點必須是硬擋，不是警告。** 實測目前 3 個
  `stage=scenes` 的目錄全部是 v3（vs 現行 v4），若只警告，使用者很可能忽略警告點下去，付出一次
  完整重跑的成本卻誤以為只是接續。緩解：UI 偵測到 `schema_version` 不符時直接把選擇重設為
  `None`，`st.error`（非 `st.warning`）說明「續跑不會沿用任何已完成階段，會從拆書重新生成」。
- **[取捨] `library.py` 的刪除操作需要一個路徑越界防呆。** `load_ad_hoc()` 允許使用者指定任意
  路徑瀏覽，若刪除函式沒有檢查目標路徑是否落在 `EVAL_SCRIPTS_DIR`/`BIXIA_STATE_DIR` 之內，理論上
  能被誘導刪除專案外的檔案。緩解：`delete_record()` 對不在這兩個目錄內的路徑一律 `ValueError`，
  ad hoc 載入的紀錄本身也不提供刪除按鈕（只有匯入/匯出兩個成功路徑進 `out/eval/` 之後的紀錄才有
  刪除鍵）。
- **[取捨] `eval/model_catalog.json` 與 `eval/model_prices.json` 是兩份手維護/半自動的檔案，
  可能漂移。** `scripts/refresh_prices.py` 只重新產生後者。緩解：`tests/test_catalog.py` 做雙向
  ID 集合一致性檢查（catalog 有的 model_prices 也要有，反之亦然），CI 等級的漂移守衛，而非文件
  約定。
- **[未驗證] reasoning effort 調低/關閉對劇本品質（結構化輸出穩定度、因果一致性）的實際影響。**
  本 change 只負責把旋鈕接上並記錄在執行紀錄裡，不含任何自動判斷「這個 effort 值太低」的邏輯；
  A/B 比較留給使用者透過總覽表按 `reasoning_effort` 分組觀察。

## Migration Plan

無資料遷移。`REASONING_EFFORT` 新增環境變數，未設定時的行為（`"default"`）與今日逐位元組相同。
`Variant.reasoning_effort`/`RunReport.reasoning_effort`/`RunRecord.reasoning_effort` 皆為新增、
選填、預設空字串或 `None`，既有 `eval/model_variants.json`、既有 `out/generation_runs*.jsonl`
的舊列不受影響（舊列讀取時 `reasoning_effort` 缺值視為空字串，代表「早於此欄位」）。

## Open Questions

- 匯入劇本時，`variant`/`requirement` 兩個欄位目前設計為使用者手動輸入——是否該讓匯入介面提供一個
  「從檔名反解」的預填建議（用 `review.parse_script_filename()` 猜 variant/slug）？留給實作階段依
  UX 手感決定，不影響本 change 範圍。
- 續跑檢查點若未來想支援持久化 variant/篇幅設定（決策四留下的後續），需要的 `PipelineState` schema
  擴充與 `_SCHEMA_VERSION` bump 是否併入同一次 change，還是獨立提案——留待那時候的實際需求決定。
