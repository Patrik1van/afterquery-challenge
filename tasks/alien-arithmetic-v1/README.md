# alien-arithmetic-v1

A Harbor task that tests **experimental design under a budget**. The agent must
discover six hidden binary operators by querying a "lab" (at most 10 queries),
then use what it learned to solve six operator-and-parenthesis puzzles.

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
│ /usr/local/bin/evaluate  (client) ───┼───────▶│ 10-query budget, query log    │
└──────────────────────────────────────┘        │ /var/log/lab/queries.jsonl    │
                                                └───────────────────────────────┘
                  │ artifacts: answer.json + queries.jsonl
                  ▼
┌──────────── verifier (separate image) ───────────┐
│ tests/verify.py + tests/answers.json             │
│ evaluates each answer with the true semantics    │
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
| `tests/verify.py` | Scoring: reward = puzzles solved / 6, plus diagnostics |
| `tests/answers.json` | Puzzles and reference solutions (verifier image only) |
| `solution/solve.py` | Oracle: a scripted experimenter that solves it in 8 queries |

## Prerequisites

- [uv](https://docs.astral.sh/uv/) and this repo's dependencies: run `uv sync` in the repo root (installs `harbor==0.26.0`).
- **Docker Desktop running.** If it isn't, Harbor fails with `Docker daemon is not running`. Start it with `open -a Docker`.
- For model runs: an API key for the provider (examples below use Gemini).

## 1. Sanity-check the task (no API key needed)

Run these two first, every time you change the task:

```bash
# Oracle: the reference solver. Expect reward 1.0 (6/6, 8 queries).
uv run harbor run -p tasks/alien-arithmetic-v1 -a oracle -e docker

# No-op agent: does nothing. Expect reward 0.0.
uv run harbor run -p tasks/alien-arithmetic-v1 -a nop -e docker
```

If the oracle doesn't score 1.0, the task is broken. If `nop` scores above 0, the
verifier is too lenient.

## 2. Run a real agent

```bash
export GEMINI_API_KEY='your-key'
echo ${GEMINI_API_KEY:+set}      # must print "set" before you start the run

uv run harbor run -p tasks/alien-arithmetic-v1 \
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
└── alien-arithmetic-v1__<id>/
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

`verifier/details.json` has two diagnostics. They don't affect the reward:

- `anchored_on_familiar`: the answer would be right if `⊕ ⊗ ◇ ▽` were ordinary `+ × max −`. The model assumed instead of testing.
- `assumed_commutative`: the answer would be right if `⊘`/`⊛` had their arguments swapped. The model never checked argument order.

## Design notes (why v1 looks like this)

- **v0 was too easy.** In v0 (4 commutative operators, no parentheses), Gemini 3.8 Flash scored 5/5. It probed each operator twice and brute-forced the puzzles locally.
- **v1 adds pressure on the experimental part,** because brute-forcing the puzzles is always trivial (4³ to 6³ combinations):
  - Two **non-commutative** operators (`⊘`, `⊛`). One probe can be ambiguous: `10 ⊛ 3 = 19` fits both `a + b²` and `a + 3b`.
  - 6 operators × 2 probes = 12 queries, which is more than the budget of 10, so the agent must choose its queries carefully.
  - **Parentheses** in queries and answers. Puzzles P3–P6 can't be solved left to right.
- **Puzzles are generated, not hand-picked.** Each puzzle has at most 2 solutions. Each new operator is required in two puzzles, and assuming familiar or commutative semantics leads to a wrong answer.
- **Any correct expression counts.** The verifier evaluates the answer rather than matching a reference string, so there's no "unique solution" requirement to maintain.
- **Malformed queries are free and reveal nothing.** The lab parses and validates the whole expression before computing anything.

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

## Make your own variant

1. Copy the folder: `cp -R tasks/alien-arithmetic-v1 tasks/alien-arithmetic-v2`
2. Change the semantics in `environment/lab/server.py` (`OPS`) **and** in `tests/verify.py` (`TRUE`). The two must match.
3. Regenerate puzzles that need your new operators, and update `instruction.md` and `tests/answers.json`.
4. Update `solution/solve.py` so the oracle can still identify the operators within budget.
5. Run the oracle (must be 1.0) and `nop` (must be 0.0), then a real model.
