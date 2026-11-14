# v5, budget 13: Gemini 4 Hydrogen, 25 trials

All runs of `tasks/alien-arithmetic-v5` at the default budget of 13 points, with Gemini 4 Hydrogen via opencode. There are 5 puzzle sets: the fixed puzzles and generated seeds 1–4. Each set has 5 Hydrogen trials and 1 oracle (reference) run. All runs were on 2026-11-14 and include the fix that charges overflow errors.

| Set | Oracle | Hydrogen mean | Hydrogen trials |
|---|---|---|---|
| fixed | 1.00 | 0.55 | 0.69 · 0.69 · 0.69 · 0.00 · 0.69 |
| seed 1 | 1.00 | 0.41 | 0.44 · 0.56 · 0.25 · 0.44 · 0.38 |
| seed 2 | 1.00 | 0.38 | 0.56 · 0.56 · 0.25 · 0.25 · 0.25 |
| seed 3 | 1.00 | 0.26 | 0.31 · 0.31 · 0.31 · 0.13 · 0.25 |
| seed 4 | 1.00 | 0.45 | 0.13 · 0.69 · 0.69 · 0.38 · 0.38 |
| **all** | **1.00** | **0.41** (sd 0.20) | 0 of 25 perfect, 0 timeouts |

How often each operator was committed correctly (24 trials committed a model; one stopped early without one):

| ⊕ | ⊗ | ◇ | ▽ | ⊘ | ⊛ | ⊞ | ⊠ | △ |
|---|---|---|---|---|---|---|---|---|
| 8 | 23 | 21 | 8 | 24 | 23 | **0** | 21 | 13 |

## Contents

| Path | What it is |
|---|---|
| `runs.tsv` | Label → original Harbor job folder (`jobs/` is gitignored) |
| `trials/<set>/<trial>/` | Per-trial evidence: `result.json`, `trial.log`, `exception.txt` (if any), `verifier/details.json`, `artifacts/` (answer, query log, committed model, posed puzzles), and `agent/trajectory.json` (the full conversation: messages, reasoning, tool calls and outputs) |
| `oracle/<set>/<trial>/` | The same for the reference run on that set |
| `trials.json` / `trials.csv` | One row per Hydrogen trial: reward, committed model, wrong operators, false "none"s and false solutions, answers inconsistent with the model, queries, multi-operator queries, minutes to reveal, steps, cost, used Python |
| `queries.csv` | Every lab query (286): set, trial, order, expression, result, cost, budget used so far |
| `puzzles.csv` | Every graded answer (200): puzzle numbers and target, solvable?, given answer, correct?, false "none", false solution, consistent with own model, reference solutions |
| `operators.csv` | One row per trial × operator (225): true rule, committed formula, correct?, and which discriminating probes were run (`a<b`, `a>b`, `sum>=256`, negative input, `n>=3` operands) |
| `scripts/aggregate.py` | Rebuilds everything above from the job folders |
| `scripts/analyze.py` | Prints the summary statistics and a per-trial breakdown |

Probe columns (`tested_*`, `probe` in `trials.json`) describe the actual operand values each operator received. They are computed from the logged queries, including nested sub-expressions, so `(5 ▽ 19) △ 1000` counts as testing `▽` with a < b, even though the outer `△` hides the result. Read them as "a test was attempted", not "a test was informative".

## Regenerate

```bash
python3 results/v5-budget13-hydrogen/scripts/aggregate.py   # needs the original jobs/ folders listed in runs.tsv
python3 results/v5-budget13-hydrogen/scripts/analyze.py     # summary + per-trial breakdown
```

The original jobs also contain opencode's internal database and raw stream (`agent/opencode/`, `agent/opencode.txt`). These are left out here because `trajectory.json` holds the same conversation.
