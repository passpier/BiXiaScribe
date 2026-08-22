## Context

見 `proposal.md` — Why。本節記錄支撐設計決策的實測數字，以及三處刻意不照
`武俠劇本資料庫Schema設計.md` 照單全收的地方（管線硬需求，非新的設計選擇）。

### 證據一：wire schema 尺寸

`wire.lenient_mirror(M).model_json_schema()`（即 `crewai.utilities.converter.
generate_model_description()` 實際送給 provider 的形狀，`ensure_all_properties_required()` 讓
每個欄位都變成 required，與 `schema.py` 自己的 default 無關）：

| model | 現況 | 改完 | |
|---|---|---|---|
| `Script` | 13,632 | 8,016 | −41% |
| `Event` | 4,039 | 2,445 | −39% |
| `ExtractionResult` | 8,510 | 4,794 | −44% |
| `BeatSheet` | 2,002 | 1,785 | −11% |
| 合計 | 28,183 | 17,040 | −40% |

每物件必填欄位數：`Event` 11→10、`Branch`/`Choice` 9→7、`SkillCheck`/`Check` 5→3、`NPC` 10→6。

### 證據二：Phase 3 的線上實測基準

`.bixia_state/1787309292-req-d232acf2d8`（`deepseek-v4-flash-0731`，六 role 同一模型，layered，
`script_length=long`）：`call_elapsed_s / elapsed_s ≈ 99.99%`（guardrail 重試、因果修復、結構化降級
三者皆為 0），completion token 中 reasoning token 占比 ~71%。時間幾乎全落在「那一次成功的 LLM
呼叫本身」——減少 wire-required 欄位數量是直接壓在這條路徑上的槓桿，不是間接優化。

### 三處無法照文件照單全收

1. **`Choice.effects: str` 保留。** `causal.py::event_to_node()` 的 `PlotNode.postconditions`
   唯一來源是 `branches[*].effects`（`causal.py:263`）；2026-08-21 瘦身已把它列為「唯一不能犧牲的
   品質底線」。這次把 precondition 端也保住（`Event.preconditions`），若 postcondition 端只剩
   `delta: int`（純數字，無法與 precondition 文字比對），`check_scene_consistency()` 一樣是永久
   no-op——保 precondition 不保 postcondition 沒有意義。
2. **`Event.summary`/`.title` 保留。** `context_builder.py::_scene_summary()` 把 `summary` 組成
   `SessionDocument.scene_summaries`，是 layered 管線裡每一場戲唯一知道前面發生過什麼的管道；
   刪掉等於管線失去跨場次記憶。`title` 供 `review.event_titles()`/UI/
   `metrics.continuity_metrics` 的 `distinct_event_title_pct` 使用。`Event.location` 則照文件
   刪除，地點收斂到 `Chapter.loc`（文件的「線性地點鏈」原則）。
3. **`Chapter.summary` 保留**（`context_builder._chapter_card()` 用）。

`Script.premise` 依文件消失（併入 `meta.theme`）；`Outline.premise` 保留為 layered 管線內部欄位，
`_assemble_script` 時映射進 `meta.theme`。

## Goals / Non-Goals

**Goals:**
- 把 `Script`/`ExtractionResult`/`Event`/`BeatSheet` 的 wire schema 縮減到上表的數字，直接減少每次
  結構化輸出呼叫要吐的必填欄位數。
- 保留因果一致性檢查（`check_scene_consistency`）的輸入輸出兩端，不讓它退化成永久 no-op。
- 修掉 `check_script_rpg`/`check_extraction_rpg` 要求 `stats>=2` 與 `check_single_stat`
  要求恰好 1 的既有矛盾（改成單一 `Stat` 物件後自然消失）。
- 清掉三處已無 schema 依據卻還在燒 token 的 dead prompt clause。
- 移除確認不可行的 `flash-glm-prose` eval 變體。

**Non-Goals:**
- 不做欄位相容層／雙寫（old+new 並存）——這是一次性 breaking rename，中途測試紅是預期的。
- 不轉檔既有 `out/eval/*.json`／`.bixia_state/` 舊資料；接受欄位靜默消失。
- 不重新設計 `region`/`sub_location`/`quest`（2026-08-21 已刪，這次不复議）。
- 不驗證真實模型呼叫是否變快（那是後續花錢驗證的動作，見 proposal.md「Cost」與 tasks.md 最後
  一節），本 change 只保證離線結構正確與 wire schema 尺寸如實縮小。

## Decisions

**決策一：全採用文件欄名（含 breaking rename），不做「只刪欄位、保留欄名」的折衷。**
理由：只刪欄位可以再省 ~12pp（35% vs 47%），但欄名不變不會讓 prompt 更短——`tasks.py` 裡的
`stat_thresholds`/`effect_ops`/`immediate_feedback` 等中文說明文字，長度跟欄位存在與否無關，跟
「要不要在 prompt 裡解釋這個概念」有關。改欄名的邊際成本（~12 個測試檔 + UI + FakeLLM）一次性
付清，換來的是欄位語意也同步變簡單（`check.on_pass` 比 `SkillCheck.success_next_event_id` 更容易
讓模型正確填寫），而不只是欄位變少。

**決策二：`Event.triggers` 改名為 `Event.preconditions: list[str]`，不直接刪除。**
理由：`causal.py:262` 是 `PlotNode.preconditions` 的唯一來源；直接刪除會讓
`CAUSAL_VALIDATION`（預設 `repair`，會真的發 LLM 修復 task）永遠找不到衝突可修，變成「看似有在
檢查，實際上已經關閉」的假象，比明確關掉 `CAUSAL_VALIDATION=off` 更危險。`list[Trigger]`（每個
含 type+condition 兩個 wire-required 欄位）→ `list[str]`，語意不變、wire 更小。

**決策三：NPC 保留 `personality`/`speech_style`，不縮到文件的 `id/name/faction_id/role` 四欄。**
理由：`speech_style` 是對話 agent RAG prompt 的直接輸入（決定武俠語感），`personality` 影響角色
台詞的一致性——這兩個欄位就是「劇本品質」本身，不是可有可無的裝飾欄位，跟文件砍掉的
`stat_threshold`/`effect_ops`（給多變數引擎用、本專案從未真正需要）性質不同。

**決策四：舊 `out/eval/*.json`／`.bixia_state/` 接受讀取失敗，不做轉檔。**
理由：這次是 breaking rename（不只刪欄位），`meta` 是全新的必填欄位、`DialogueLine.npc`
取代了必填的 `npc_id`，所以舊檔案在 `Script.model_validate()` 會直接拋
`ValidationError`，不是「多的欄位被 `extra="ignore"` 悄悄丟掉」這種溫和降級（那是給
*刪除選填欄位*用的說法，這次連 required 欄位的名字都變了）。這正好是 UI 端既有的
「無法讀取此劇本檔案」優雅降級路徑（`ui/app.py:93-104`，`_load()` 捕捉任何 exception 回傳
`None`）設計時就準備要接住的情況——`overview_rows()`/`_load()` 兩處都已經是
try/except 包住 `load_script()`，不需要新程式碼。轉檔 12 份 `out/eval/*.json` 與 11 個
`.bixia_state/` run 目錄的成本，遠高於它們作為「舊 A/B 比較基準」的殘餘價值。

**決策五：validate/guardrail 刪除而非保留空殼。**
`validate_stat_thresholds()`/`validate_npc_introductions()`/`validate_truth_layering()` 與五個
guardrail（`check_delayed_payoff`/`check_stat_narrative`/`check_single_stat`/`check_scene_mix`/
`check_convergence`）直接刪除函式本體，不留一個「永遠回傳 []」的空殼。理由：空殼會讓
`collect_quality_problems()` 看起來還在做 9 項檢查，實際上只剩 4 項——這跟決策二否決的「假象比
明確關閉更危險」是同一個原則。

## Risks / Trade-offs

- **[取捨] 收斂（converge）概念整組消失。** `chapter.converge_event_id` +
  `branch.converges_to_event_id` + `check_convergence` 全刪，文件用 `choices[].payoff_at`
  取代。副作用：`normalize.py::_fix_next_event_ids` 的 `next_event_id` 回填 fallback 鏈
  少了兩層（`converges_to_event_id` 與 `chapter.converge_event_id`），只剩「事件序列中下一個
  event」。**接受**，`_fix_next_event_ids` 的 note 字串要誠實反映只剩一層，不要保留舊字串誤導
  「有嘗試過 converge 回填」。
- **[風險] `Event.preconditions: list[str]` 比 `list[Trigger]` 更容易被模型留空**——舊的
  `type`/`condition` 兩個 wire-required 欄位某種程度上「逼」模型思考觸發條件，改成單一
  `list[str]` 後模型更容易交空陣列。緩解：scene_write prompt 明確要求「至少寫出這場戲成立的前提
  一句話」，並在 `check_scene_rpg` guardrail 加一條 `preconditions` 非空檢查（本 change 唯一新增
  的 guardrail，其餘全是刪除/改寫）。
- **[風險] 少了 `stat_thresholds`，`endings[].min/max` 可能區間重疊或留下缺口。** 緩解：在
  `validate_references()` 之外加一個純函式（`check_ending_ranges` 或併入
  `check_choice_quality` 附近）檢查 endings 區間不重疊，report-only，接到
  `collect_quality_problems()`，沿用既有慣例。
- **[風險] 這是一次改 20+ 檔案的 breaking rename，過程中測試必然是紅的。** 緩解：按
  `tasks.md` 的順序（schema → causal → guardrails → prompts → context_builder →
  orchestrator/normalize/metrics → llm.py → review/ui → tests → docs）逐步推進，每完成一個大區塊
  跑一次 `pytest tests/`，不要求中途保持全綠。
- **[未驗證] schema 變小是否真的讓真實模型呼叫變快。** Phase 2 同樣的問題到現在都還沒花真錢驗證
  過。本 change 的驗證計畫見 `tasks.md` 最後一節：只跟 `flash-only` 變體（唯一目前通過測試、也是
  正式環境實際配置）比較，不使用已確認不可行的 `flash-glm-prose`，也不使用不同模型量級的
  `deepseek-v4-pro`/`baseline`（會混淆變因）。

## Migration Plan

`_SCHEMA_VERSION` 3 → 4（`crew/orchestrator.py`）——正在跑的舊 checkpoint 直接判定為「沒有
checkpoint」重新開始，不嘗試遷移，與前兩次 bump（1→2、2→3）同一慣例。已產出的
`out/eval/*.json`／`.bixia_state/*/script.json` 不受影響，讀取時已刪欄位被
`extra="ignore"` 忽略；不做批次轉檔。

## Open Questions

- `endings[].min/max` 的重疊/缺口檢查該是新函式還是併入 `check_choice_quality`？留給實作階段
  依程式碼位置就近決定，不影響本 change 的範圍。
- `check_scene_rpg` 新增的 `preconditions` 非空檢查該是 hard guardrail（進 in-loop 重試）還是
  report-only？傾向 report-only（沿用其餘九項離線檢查的慣例，避免又製造一組新的重試預算消耗），
  但若離線測試發現模型持續留空導致因果檢查形同虛設，可在實作階段升級為 hard guardrail。
