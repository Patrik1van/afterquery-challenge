# Notable results: reference vs Gemini 4 Hydrogen vs Gemini 4 Carbon

Falsify, budget 20, puzzle set `FALSIFY_SEED=42`, 5 trials per model, 60-minute limit per trial.

| Run | Mean reward | Perfect | Operators right | Time / trial |
|---|---|---|---|---|
| Reference solver | **1.00** | 1/1 | 9/9 | ~3 min |
| Gemini 4 Carbon | **0.78** | 1/5 | 7–9 | 28–35 min |
| Gemini 4 Hydrogen | **0.73** | 0/5 | 7–8 | 3–7 min |

**Key findings**
- **`⊞ = (a + b) mod 256` decides the gap.** It was wrong in 9 of 10 model trials, and every 0.81 is "all correct except `⊞`". The failures took three forms:
  - **Never probed a sum ≥ 256:** all 5 Hydrogen trials and 2 Carbon trials.
  - **Probed it, but hid the result:** `20 △ (410 ⊘ (181 ⊞ 319)) = 0`, where the outer `△` returns 0 either way.
  - **Explained the wraparound with a wrong rule:** after `… ⊞ 601 → 89`, the agent committed `a * i + b // i` instead of modulo.
- **Only one model trial found the modulo** (Carbon, via `(16 ⊠ 28) ⊛ (410 ⊞ 504)`), and it scored 1.0.
- **The reference knows only the instruction's grammar,** with no candidate list. It needs at least **19 points** to identify all nine operators (`reference_budget_sweep/`), so budget 20 is fair.
- **Carbon probes more carefully than Hydrogen but gains little.** It probed `⊞` beyond 256 in 3 of 5 trials against 0 of 5, yet scored only +0.05 higher, while taking about 5× longer.

**Contents**
- `trials/<run>/<trial>/`: the full model transcript (`agent/trajectory.json`, or `agent/oracle.txt` for the reference), the query log, the committed model, the answers and `verifier/details.json`.
- `trials.csv`: one row per trial.
- `queries.csv`: every lab query.
- `puzzles.csv`: every graded answer.
- `operators.csv`: each trial's committed formula per operator, whether it's correct, and which tests were run.

These runs were recorded before the task was renamed from `alien-arithmetic-v5`, so file contents use the old name.
