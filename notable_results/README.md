# Notable results: reference vs Gemini 4 Hydrogen vs Gemini 4 Carbon

Falsify, budget 20, 5 trials per model, 60-minute limit per trial.

## Across seeds: Hydrogen is consistently below the reference

| Seed | Reference | Gemini 4 Hydrogen (5 trials) | Hydrogen mean |
|---|---|---|---|
| 42 (default) | 1.00 | 0.81 · 0.38 · 0.81 · 0.81 · 0.81 | 0.73 |
| 7 | 1.00 | 0.81 · 0.50 · 0.50 · 0.50 · 0.50 | 0.56 |
| 2026 | 1.00 | 0.81 · 0.81 · 0.44 · 0.69 · 0.44 | 0.64 |
| **All** | **1.00** | **0 of 15 perfect, best 0.81** | **0.64** |

The reference solved every seed perfectly. Hydrogen got `⊞` wrong in all 15 trials, which caps it at 0.81.

## Seed 42: reference vs Hydrogen vs Carbon

| Run | Mean reward | Perfect | Operators right | Time / trial |
|---|---|---|---|---|
| Reference solver | **1.00** | 1/1 | 9/9 | ~3 min |
| Gemini 4 Carbon | **0.78** | 1/5 | 7–9 | 28–35 min |
| Gemini 4 Hydrogen | **0.73** | 0/5 | 7–8 | 3–7 min |

`⊞ = (a + b) mod 256` decides the gap: it was wrong in 9 of 10 model trials, and every 0.81 is "all correct except `⊞`". The only model trial that found the modulo (Carbon, via `(16 ⊠ 28) ⊛ (410 ⊞ 504)`) scored 1.0.

## Where the models fail

Each pitfall is labelled with the data it comes from. **[seed 42]** is the seed-42 runs in this folder. **[budget 13]** is an earlier set of 25 Hydrogen trials on five puzzle sets with 13 points, which is not included here.

1. **Stays in the comfort zone: tries small, typical inputs and never pushes to the edges.** **[seed 42]**
   - Every first probe used operands of 38 or less.
   - 7 of 10 trials never tested `⊞` with a sum ≥ 256 (Hydrogen 0 of 5), although operands up to 1000 are allowed.
2. **Stops at the first fit: a pattern that looks right is accepted without a test.** **[seed 42]**
   - 7 trials committed `⊞ = a + b` after seeing only small sums such as `17 ⊞ 6 = 23`.
   - One Carbon trial saw `… ⊞ 601 → 89` (601 mod 256) and committed the first rule that fit, `a * i + b // i`, without testing it.
3. **Wastes experiments: runs checks whose outcome can't change its mind.** **[seed 42]**
   - `20 △ (410 ⊘ (181 ⊞ 319)) = 0`: the outer `△` returns 0 whatever `⊞` does.
   - In those 7 trials, the second `⊞` probe again used a small sum (e.g. `3 ⊞ 29`), so it couldn't reveal the wraparound either.
4. **Trusts the task over its data: bends its own model to make a puzzle solvable.**
   - **[seed 42]** Hydrogen hyarm73 committed `▽ = a − b`. After the reveal it switched to `abs(a − b)` because only that made P2 solvable: *"This observation provides strong evidence that the `▽` operator is likely `abs(a - b)`."*
   - **[budget 13]** 10 of 25 trials submitted answers that contradict their own committed model. One example: *"`⊘ = n*a - b` yielded 8/8 solutions but `⊘ = 2*a - b` only 6/8."*
5. **Makes up answers to avoid the penalty: a wrong `"none"` costs ½ and a wrong expression costs nothing, so it invents a solution instead of saying there is none.**
   - **[budget 13]** 8 answers in 5 of 25 trials gave an expression for an unsolvable puzzle that even the agent's own model says doesn't reach the target. The reasoning behind them reads like *"comparing the expected values of `"none"` versus an expression"*. All 8 scored 0.
   - **[seed 42]** Not observed. hyarm73 noticed the asymmetry (*"There's no penalty for incorrect expressions, but a half-point penalty for 'none'"*) but still answered `"none"` where its model found no solution.

## Other findings

- **The reference is fair.** It knows only the instruction's grammar, with no candidate list, and needs at least **19 points** to identify all nine operators (`reference_budget_sweep/`). Budget 20 leaves one point of slack.
- **Carbon probes more carefully than Hydrogen but gains little.** It probed `⊞` beyond 256 in 3 of 5 trials against 0 of 5, yet scored only +0.05 higher, while taking about 5× longer.

## Contents

- `trials/<run>_seed<N>/<trial>/`: the full model transcript (`agent/trajectory.json`, or `agent/oracle.txt` for the reference), the query log, the committed model, the answers and `verifier/details.json`.
- `trials.csv`: one row per trial.
- `queries.csv`: every lab query.
- `puzzles.csv`: every graded answer.
- `operators.csv`: each trial's committed formula per operator, whether it's correct, and which tests were run.

These runs were recorded before the task was renamed from `alien-arithmetic-v5`, so file contents use the old name.
