# Falsify

**Motivation.** Scientific work means testing hypotheses under a limited experiment budget: listing the explanations that fit, choosing the experiment that tells them apart, and committing only as far as the evidence goes. This task measures whether an agent can do that, rather than just reason over information it was handed.

**Two phases.**
1. **Experiment.** The agent probes nine unknown binary operators through a lab it can only reach over HTTP. It has 20 budget points, and each query costs one point per operator. Each operator looks familiar under a naive probe but isn't (for example, `⊞` is `(a + b) mod 256`). The agent then commits a formula for every operator.
2. **Puzzles.** The lab stores the model, locks itself, and only then reveals 8 puzzles: place operators and parentheses so that `n1 ? n2 ? n3 ? n4 = target`. Some puzzles have no solution, and the correct answer for those is `"none"`.

**Reward:** `max(0, correct − 0.5 × wrong "none") / 8`. Doing nothing, answering `"none"` everywhere, or committing the obvious model without experimenting all score 0.

## Results (seed 42, budget 20, 5 trials per model)

| Run | Mean reward | Perfect runs |
|---|---|---|
| Reference solver | **1.00** | 1/1 |
| Gemini 4 Carbon | 0.78 | 1/5 |
| Gemini 4 Hydrogen | 0.73 | 0/5 |

**Across seeds (Hydrogen, 5 trials each):**

| Seed | Reference | Hydrogen |
|---|---|---|
| 42 | 1.00 | 0.73 |
| 7 | 1.00 | 0.56 |
| 2026 | 1.00 | 0.64 |
| **All** | **1.00** | **0.64** (0/15 perfect) |

Details: [notable_results/](notable_results/).

The reference solver (`tasks/falsify/solution/solve.py`) knows only what the agent is told: it enumerates the instruction's formula grammar, prefers shorter formulas, and picks the most informative query per point. Both models mostly fail on `⊞`: they never probe a sum ≥ 256, probe it in a way where another operator hides the result, or explain the wraparound with a wrong rule.

## Reward integrity

- **Baselines score 0.0** (seed 42): doing nothing, answering `"none"` everywhere, or committing familiar operators without experimenting.
- **Stopping early scores 0** if no answers were written, since `/app/answer.json` starts as `{}`. Answers written before the stop count only if correct.
- **The budget is enforced by the lab:** a single-threaded server, so retries and parallel requests can't bypass it. Malformed or unaffordable queries are refused for free. Evaluated queries are charged, even if they overflow.
- **No hidden state leaks:** the true operators exist only in the lab and verifier images, and the puzzles appear only after the model is committed and the lab is locked.
- **Grading:** any expression that reaches the target is accepted. A wrong `"none"` costs ½, and an expression given for an unsolvable puzzle scores 0.

## Run

Requires Docker and `uv sync` in the repo root. From the repo root:

```bash
# Reference vs Gemini 4 Hydrogen vs Gemini 4 Carbon
GEMINI_API_KEY=... tasks/falsify/run_comparison.sh

# Reference vs Gemini 4 Hydrogen only
GEMINI_API_KEY=... tasks/falsify/run_comparison.sh --quick
```

Options: `--seed N` (default 42) and `--trials K` (default 5). The script prints a summary table, and per-trial diagnostics are in `jobs/<run>/<trial>/verifier/details.json`.

**Time.** Each trial is limited to **60 minutes**. Carbon takes about 30–35 minutes per trial, so the full comparison can take up to about an hour. `--quick` is much faster: the reference takes about 3 minutes and Hydrogen about 5 minutes per trial, so the whole run takes roughly 10–15 minutes including container setup.

Single runs:

```bash
uv run harbor run -p tasks/falsify -a oracle -e docker   # reference → 1.0
uv run harbor run -p tasks/falsify -a nop -e docker      # do nothing → 0.0
```

The lab generates its puzzles from `FALSIFY_SEED` (default 42, the set behind the reported results); set `FALSIFY_SEED=N` for another puzzle set. The verifier computes solvability by brute force, so every puzzle set is graded the same way.

## Files

| Path | Purpose |
|---|---|
| `tasks/falsify/instruction.md` | The prompt the agent sees |
| `tasks/falsify/environment/lab/server.py` | Lab sidecar: hidden operators, budget, query log, commit and lock |
| `tasks/falsify/environment/lab/generator.py` | Seeded puzzle generator |
| `tasks/falsify/environment/evaluate`, `reveal` | The agent's command-line tools |
| `tasks/falsify/tests/verify.py` | Separate verifier: reward and diagnostics |
| `tasks/falsify/solution/solve.py` | Reference solver |
| `tasks/falsify/run_comparison.sh` | Reference vs Gemini comparison |
