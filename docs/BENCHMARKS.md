# BiXiaScribe — 關鍵數據

這份文件集中收錄本專案所有實測結果（檢索品質、生成成本/品質 A/B）。README 只留三到五行重點結論
加連結；「為什麼這樣設計」與「怎麼跑」看 [`DESIGN_NOTES.md`](./DESIGN_NOTES.md)。每組數據標了
量測日期與指令，都可以用同一個指令重跑覆核。

## 1. 檢索：hybrid vs 純向量

`scripts/eval_retrieval.py`，14 部金庸全集 + 11 本 capped webnovel 索引，`eval/retrieval_eval.jsonl`
的 14 條武俠查詢（12 條有 ground truth）。實跑於 2026-07-29，本機 `bge-m3`。

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

## 2. 生成：no-RAG A/B（2026-08-19，最新一輪）

`SCRIPT_LENGTH=long`，`--pipeline-mode legacy`，n=5/組，
`scripts/eval_generation.py --variants deepseek-v4-pro,deepseek-v4-pro-norag`。這組兩個變體
模型組合完全相同（機械角色 extractor/beat_expander 用 deepseek-v4-flash-0731，
writer/dialogue-scene_writer/proof 用 deepseek-v4-pro-0423），唯一差異是 `use_retrieval`：

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

## 3. 生成：四組模型組合 A/B（較早一輪）

`scripts/eval_generation.py --pipeline-mode legacy --script-length medium`，n=5/variant。這是
比第 2 節更早的一輪 A/B，測的是四組完全不同的模型組合（不只是有/無檢索），留存作為方向性參考：

| Variant | 成功率 | avg events | avg 對話行長 | avg retrieval_calls | avg tokens | avg cost/run |
|---|---|---|---|---|---|---|
| baseline（全用 deepseek-chat） | 5/5 | 9.4 | 19.3 字 | 2.40 | 26,914 | $0.0137 |
| long-cheap（全用 deepseek-v4-flash-0731，Decart） | 5/5 | 16.2 | 43.0 字 | 10.40 | 110,398 | $0.0081 |
| long-prose（dialogue/scene_writer → glm-5.2，Novita） | 5/5 | 15.8 | 55.2 字 | 11.60 | 137,765 | $0.0099 |
| long-mimo（全用 xiaomi/mimo-v2.5，GMICloud） | 1/2 | 7.0 | 64.3 字 | 11.00 | 180,159 | $0.0244 |

`long-cheap` 產出近 2 倍 baseline 的事件數，成本卻低 40%，`retrieval_calls` 更是 baseline 的
4 倍——當時是取代 `baseline` 的最強候選。`long-prose` 把 dialogue/scene_writer 換成
`z-ai/glm-5.2`，成本略增，但肉眼讀 `out/eval/*.json` 是四組裡武俠語感最好的（句子更長、更自然，
不像片段）。`long-mimo` 不推薦——實際跑不穩定（provider 偶爾回傳 `choices=None`，或模型會幻覺出
不存在的 schema 欄位），且是四組裡最貴的。

**已知限制**：`long-cheap`/`long-prose`/`long-mimo` 各自 pin 了特定 OpenRouter provider
（`LLM_PROVIDER_ONLY` 是行程層級 env var，見 `CLAUDE.md`），而 `--pipeline-mode layered` 在它們
pin 的 endpoint 上會 unhandled crash（crewai 收到 `choices=None` 回應，沒被包成
`PipelineError`）——這張表完全是在 `--pipeline-mode legacy` 下量測的，這三組變體目前在
`layered` 模式下不可用。

## 4. 非顯而易見的發現

`retrieval_calls` 顯示：「模型宣稱支援 function calling」（OpenRouter `/models` metadata）
不等於「在 CrewAI 的 ReAct loop 裡真的會主動呼叫工具」——需要逐模型檢查 `retrieval_calls`，
不能只看 provider 標示的能力。完整分析方法見
[`DESIGN_NOTES.md`](./DESIGN_NOTES.md#4-比較不同-agent-的模型組合)；逐句台詞比較需要肉眼讀
`out/eval/` 下實際存的劇本 JSON（用 UI 的並排比較模式，見 [README](../README.md)）。

## 5. 成本回顧與重跑方式

`src/bixiascribe/pricing.py` 會對每一列精確計算 `cost_usd`（含 prompt cache 折扣，見
`cost_basis`）。`python scripts/eval_generation.py --dry-run` 會在花費任何 token 前印出完整
矩陣的預估成本。

```bash
# 檢索品質（vector vs hybrid）
python scripts/eval_retrieval.py
python scripts/eval_retrieval.py --top-k 1   # 嚴格比較，差異更明顯

# 生成 A/B：先零成本估價，再真的跑
python scripts/eval_generation.py --dry-run
python scripts/eval_generation.py --variants deepseek-v4-pro,deepseek-v4-pro-norag --repeat 1

# 只想重新看彙總表，不想再花錢
python scripts/eval_generation.py --from-jsonl out/generation_runs.jsonl
```

方法論細節（如何設計 A/B、如何跨 provider 比較）見
[`DESIGN_NOTES.md`](./DESIGN_NOTES.md#4-比較不同-agent-的模型組合)。
