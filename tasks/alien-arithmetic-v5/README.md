# alien-arithmetic-v5

A Harbor task that tests **experimental design under a budget**, in two phases:

1. **Experiment.** The agent probes the lab without knowing the puzzles, then commits a formula for every operator (`/app/model.json`) with `reveal`.
2. **Puzzles.** The lab stores the model, **locks itself** and only then hands out the puzzles. No more experiments are possible.
 The agent must
discover six hidden binary operators by querying a "lab" (a budget of 8 points,
where each query costs one point per operator in it),
then use what it learned to solve eight operator-and-parenthesis puzzles.
Two of the puzzles have no solution, and the agent isn't told which ones or
how many. It must answer `"none"` for those, so it has to be confident in
its picture of the operators rather than just consistent with the puzzles.

It's a worked example of three patterns you can reuse in your own tasks:

1. **Hidden state behind a sidecar.** The rules live in a separate `lab`
   container that the agent can only reach over HTTP, so they can be probed
   but never read.
2. **A budget enforced on the server.** The agent can't bypass the query limit
   by retrying, racing parallel requests or editing a client.
3. **A separate verifier.** The expected answers exist only in the verifier
   image, and the evidence (answer + query log) is copied in as artifacts.

## How it works

```
┌──────────── main (agent) ────────────┐        ┌──────── lab (sidecar) ────────┐
│ /app/answer.json   (agent writes)    │  HTTP  │ server.py: hidden operators,  │
│ /usr/local/bin/evaluate  (client) ───┼───────▶│ 13-point budget, query log    │
└──────────────────────────────────────┘        │ /var/log/lab/queries.jsonl    │
                                                └───────────────────────────────┘
                  │ artifacts: answer.json, queries.jsonl, model.json, puzzles.json
                  ▼
┌──────────── verifier (separate image) ───────────┐
│ tests/verify.py + tests/semantics.json           │
│ brute-forces the posed puzzles, grades answers   │
│ → /logs/verifier/reward.txt, details.json        │
└──────────────────────────────────────────────────┘
```

| Path | What it is |
|---|---|
| `instruction.md` | The prompt the agent sees (operators, rules, puzzles, answer format) |
| `task.toml` | Timeouts, resources, artifacts, separate-verifier config |
| `environment/Dockerfile` | The agent's container (no secrets in it) |
| `environment/docker-compose.yaml` | Adds the `lab` sidecar next to Harbor's `main` service |
| `environment/lab/server.py` | Hidden semantics, expression parser, budget, query log |
| `environment/evaluate` | The CLI the agent uses: `evaluate "2 ⊕ (3 ⊘ 1)"` |
| `environment/reveal` | Commits the model, locks the lab, prints the puzzles and saves them to `/app/puzzles.json` |
| `environment/lab/puzzles.json` | The puzzles (numbers + targets only). They exist only in the lab until `reveal` |
| `environment/lab/formula.py`, `tests/formula.py` | Safe formula parser (AST whitelist, no `eval`). The two copies must be identical |
| `tests/verify.py` | Scoring: reward = puzzles answered correctly / 8, plus diagnostics |
| `environment/lab/semantics.json`, `tests/semantics.json` | The operators: each symbol's true rule and its naive reading. The two copies must be identical |
| `environment/lab/generator.py` | `generate(semantics, seed, difficulty)` draws a puzzle set; see **Generated puzzles** |
| `solution/solve.py` | Oracle: a scripted experimenter that solves it in 8 queries |

## Prerequisites

- [uv](https://docs.astral.sh/uv/) and this repo's dependencies: run `uv sync` in the repo root (installs `harbor==0.26.0`).
- **Docker Desktop running.** If it isn't, Harbor fails with `Docker daemon is not running`. Start it with `open -a Docker`.
- For model runs: an API key for the provider (examples below use Gemini).

## 1. Sanity-check the task (no API key needed)

Run these two first, every time you change the task:

```bash
# Oracle: the reference solver. Expect reward 1.0 (8/8, 9/9 model, exactly 13 points).
uv run harbor run -p tasks/alien-arithmetic-v5 -a oracle -e docker

# No-op agent: does nothing. Expect reward 0.0.
uv run harbor run -p tasks/alien-arithmetic-v5 -a nop -e docker
```

If the oracle doesn't score 1.0, the task is broken. If `nop` scores above 0, the
verifier is too lenient.

## 2. Run a real agent

```bash
export GEMINI_API_KEY='your-key'
echo ${GEMINI_API_KEY:+set}      # must print "set" before you start the run

uv run harbor run -p tasks/alien-arithmetic-v5 \
  -a opencode -m google/gemini-3.8-flash \
  -e docker -k 5 -n 5
```

- `-k 5`: 5 attempts (trials). `-n 5`: run all 5 at once (the default is 4, which makes the 5th wait).
- If the key isn't picked up, pass it explicitly: `--ae GEMINI_API_KEY="$GEMINI_API_KEY"`.
- Other agents and models work the same way, e.g. `-a claude-code -m anthropic/<model>` with `ANTHROPIC_API_KEY`.

## 3. Read the results

Each run writes to `jobs/<timestamp>/`, with one folder per trial:

```
jobs/2026-11-14__15-07-49/
├── result.json                         # job summary (mean reward, pass@k)
└── alien-arithmetic-v5__<id>/
    ├── agent/opencode.txt              # full agent transcript: what it ran and saw
    ├── artifacts/app/answer.json       # the answer it submitted
    ├── artifacts/var/log/lab/queries.jsonl   # every query it spent, in order
    ├── verifier/details.json           # per-puzzle grading + diagnostics
    └── exception.txt                   # only if the trial crashed
```

Quick per-trial summary:

```bash
J=jobs/$(ls -t jobs | head -1)
for d in $J/alien*/; do echo "$(basename $d): $(cat $d/verifier/test-stdout.txt)"; done
```

Or browse everything in the UI: `uv run harbor view jobs`.

`verifier/details.json` has these diagnostics. They don't affect the reward:

- `false_none`: answered `"none"` on a puzzle that has a solution. Usually a wrong picture of some operator.
- `false_solution`: gave an expression on a puzzle that has no solution. The model "solved" it under a wrong belief.

- `naive_reading`: the answer would be right under the naive readings in `semantics.json` (no mod, a+b+2, untested
  argument order, ...). The model assumed instead of testing.

## Design notes (how we got to v2)

| Version | Change | Gemini 3.8 Flash (5 trials) |
|---|---|---|
| v0 | 4 commutative operators, left-to-right only, 10 queries | 5/5 solved everything |
| v1 | + 2 non-commutative operators (`⊘` = 2a−b, `⊛` = a+b²), parentheses, 10 queries | 5/5 solved everything, 10/10 queries each |
| v2 | + 2 unsolvable puzzles (answer `"none"`), budget cut to 8 | 0.30 (900 s, 3 timeouts) · 0.175 (1800 s) |
| v2 | same, **Gemini 4 Hydrogen** | 1.00. Probed large asymmetric pairs, then checked `◇`/`▽` reversed inside one 3-operator query |
| v3 | + query cost = one point per operator (same puzzles, budget 8) | Hydrogen 1.00. No combined queries; 2 of 5 trials never tested `◇` with a < b and guessed `max` |
| v4 | + two phases: commit a model before seeing the puzzles, then the lab locks | Hydrogen 1.00, 6/6 model; all 5 trials tested `◇`/`▽` reversed before committing |
| v5 | + 3 operators: `⊞` = (a+b) mod 256, `⊠` = a+b+n (n = operands in the whole expression), `△` = a−b if a≥b else 0. Budget 13. P1–P4 kept from v4, P5–P8 new | *see below* |

**Why v5.** Each new operator looks ordinary under the most natural probe:
- `⊞` behaves like `a + b` until the sum reaches 256.
- `⊠` behaves like a constant `a + b + 2` in every 2-operand query. Only a query with 3–4 operands shows that the
  constant is the operand count, and every puzzle has 4 operands, so it adds 4 there.
- `△` behaves like `a − b` until a < b.

Budget 13 is the exact minimum for full verification: 9 base probes, ◇/▽/△ reversed, and ⊠ at a second operand
count. The oracle shares one 3-point query between ◇, ▽ and ⊠ at n = 4. The formula language has three documented context variables:
`n` (operands in the expression), `i` (operator position) and `d` (nesting depth). Only `n` matters; `i` and `d` are
decoys. In every 2-operand probe `a+b+n`, `a+b+i+1` and `a+b+d+2` give the same result, so documenting the variables
does not reveal which one ⊠ uses. Only a multi-operand probe does. The committed model is checked in every
(n, i, d) context.

**Reward:** `max(0, correct − 0.5 × wrong "none") / 8`. A `"none"` on a solvable puzzle costs half a point (stated in
the instruction), so answering `"none"` everywhere scores 0. **Agent time limit:** 60 minutes.

**Integrity baselines** (real verifier): do nothing 0 · `"none"` everywhere 0 · stop before `reveal` 0 · random
operators 0 · guessed model without experiments 0 · lazy 9 probes 0.25 · oracle 1.0. In the real harness, Hydrogen cut
off at 90 s scored 0.

**Results so far** (5 trials each, 30-minute limit, before the decoy variables): Hydrogen 0.725 and 0.81 (two runs),
Argon 0.475, Carbon 0.15. 6 of the 10 Carbon/Argon trials timed out, which is why the limit is now 60 minutes.

**Why v4.** In v2/v3 an agent could resolve an untested ambiguity by checking which reading made the puzzles
work. v4 makes it commit to a model *before* it sees any puzzle, and records that model. Two diagnostics
show what happened:
- `model_operators_correct` / `model_wrong_operators`: the committed formulas checked against the truth on a grid
  of 121 pairs (a < b, a > b, a = b, zeros, large values). A model that never tested `◇` with a < b but wrote
  `max(a, b)` still counts as correct; the diagnostic measures the outcome, not the evidence.
- `answers_inconsistent_with_model`: answers that don't follow from the agent's own committed model, i.e. it
  changed its picture of the operators after seeing the puzzles.

**Why v3.** Hydrogen solved v2 by packing several checks into one query, e.g.
`(13 ▽ 400) ⊕ (17 ◇ 1000)` tested `▽`, `◇` and `⊕` for the price of one query. v3 charges one point per
operator, so combining probes no longer saves budget. The 8 points are still enough: the oracle needs exactly 8
single-operator probes.

**Why v1 was still easy.** Brute-forcing puzzles locally is always trivial (at most 6³ × 5 bracketings), so all
the difficulty has to sit in identifying the operators. In v1 two shortcuts made that cheap:

1. *Puzzles leaked information.* We promised every puzzle had a solution. One probe `10 ⊛ 3 = 19` fits
   both `a + b²` and `a + 3b`, but under `a + 3b` one puzzle had no solution, so the model rejected that
   hypothesis for free.
2. *Spare queries for verification.* With 10 queries, a model could probe each operator once and spend the
   remaining 4 checking whole candidate answers.

**What v2 changes.**

- **Unsolvable puzzles.** "My hypothesis leaves this puzzle unsolvable" no longer proves the hypothesis wrong.
  The answer might really be `"none"`. The task now also measures calibration.
- **No uniqueness promise.** With parentheses it can't be guaranteed anyway: `⊕` is associative, so
  `(a⊕b)⊕c = a⊕(b⊕c)`. A promise would also leak the other way ("2 solutions, so my model is wrong").
- **Budget 8 for 6 operators.** That's one probe per operator plus 2 spare. The oracle needs exactly 8: it
  probes `(100, 7)`, then re-probes `(7, 100)` for `◇` and `▽` because `max`/`a` and `|a−b|`/`a−b`
  collide when a > b. So the task is solvable by careful design, with no margin for wasted probes.
- **Puzzles are traps in both directions** (generated, not hand-picked). We modelled five plausible wrong
  beliefs: familiar `+ × max −`, swapped `⊘`/`⊛`, `⊛ = a+3b`, `◇ = a`, `▽ = a−b`.
  - Each of the 6 solvable puzzles has exactly one solution, and every relevant wrong belief misses it
    (it finds a different expression or none). Each wrong belief is caught by at least 2 puzzles.
  - Each of the 2 unsolvable puzzles *is* "solvable" under 2–3 of those wrong beliefs, so a model with the
    wrong picture confidently answers an expression.
- **Scoring stays simple.** A wrong `"none"` scores 0, with no penalty. Answering `"none"` everywhere gives 0.25;
  that's the floor to compare models against. If models hedge with `"none"` too much, the next step is a
  penalty for `false_none`.

## Gotchas we hit

| Symptom | Cause / fix |
|---|---|
| `Two workspace members are both named ...` | A stray `pyproject.toml` in a parent directory (e.g. `~`). Rename or remove it. |
| `network_mode='no-network' is not supported by docker` | Docker Desktop's kernel lacks `CONFIG_NFT_FIB_INET`. Comment out `network_mode` under `[verifier.environment]` for local runs, or use a cloud env (`-e daytona` / `-e modal`). |
| `ProviderAuthError: ... API key is missing` (reward 0, 0 queries) | The key isn't in the shell running `harbor`. Check with `echo ${GEMINI_API_KEY:+set}` or pass `--ae`. |
| `export: not valid in this context` | zsh doesn't allow `#` comments in interactive commands. Remove trailing comments, or `setopt interactivecomments`. |
| Every trial spends minutes installing the agent | See **Dev speedups** below. |

## Dev speedups (remove before submitting)

`environment/Dockerfile` currently has a `DEV ONLY` block that bakes opencode into
the image, so Harbor skips the per-trial Node + npm install. It needs
`ENV NVM_DIR` and `SHELL bash`: under the default `/bin/sh`, nvm installs into
`/usr/bin/versions` and Harbor can't find it. **Remove that block, and restore
`network_mode = "no-network"` in `task.toml`, before you submit.**

## Generated puzzles

Without a seed, the lab poses the fixed v5 puzzles (`environment/lab/puzzles.json`); all reported results use them.
With a seed, it draws a new set at startup:

```bash
ALIEN_SEED=7 uv run harbor run -p tasks/alien-arithmetic-v5 -a oracle -e docker                       # difficulty 3
ALIEN_SEED=7 ALIEN_DIFFICULTY=1 uv run harbor run -p tasks/alien-arithmetic-v5 -a oracle -e docker
```

| Difficulty | Puzzles |
|---|---|
| 1 | 8 solvable, at most 2 solutions each |
| 2 | 7 solvable + 1 unsolvable; every naive reading of an operator a puzzle uses gives a wrong answer |
| 3 (default) | 6 solvable + 2 unsolvable, as in 2, and together the puzzles catch every operator's naive reading |

Unsolvable puzzles look solvable under at least two naive readings. The same seed and difficulty always give the same
puzzles. The lab writes the posed set to `/var/log/lab/puzzles.json`; the verifier grades exactly that set and computes
solvability and solutions itself by brute force, so there is no answer key to keep in sync. Only the puzzles change; the
operators, budget, rules and reward stay the same.

## Make your own variant

1. Change the operators in `semantics.json` (both copies): a `rule` and a `naive` reading per symbol.
2. Make sure `solution/solve.py` can still identify them within budget (its hypothesis library and follow-up probes).
3. Run the oracle (must be 1.0) and `nop` (must be 0.0) with and without `ALIEN_SEED`, then a real model.
