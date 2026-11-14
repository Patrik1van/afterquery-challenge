# Cold-chain Dispatch — a Harbor RL environment

The agent dispatches a small refrigerated delivery fleet through one 480-minute shift. Orders have delivery windows, capacity demands and a spoilage deadline, and **about half of them only appear mid-shift**. The agent controls the fleet with a `dispatch` CLI: look at the state, set routes, advance the clock.

## Why it's hard
- **Planning several steps ahead:** a route that looks fine now can make the next three orders spoil.
- **Adapting when new information arrives:** new orders mean replanning routes that are already moving.
- **Managing limited resources:** capacity forces depot reloads, and each reload costs 10 minutes.
- Everything the agent needs is visible through `dispatch state`. The difficulty comes from its decisions, not from hidden rules.

## Scoring (0–1)
`service = (on_time + 0.5 × late) / all_orders`, then discounted by up to 30% for distance. Every order counts, including ones the agent never looked at.

## Reward-hacking defences
| Shortcut | Why it fails |
|---|---|
| Do nothing / `finish` early | Unserved orders count as 0. The remaining shift still plays out. Idle scores 0. |
| Claim deliveries | The score comes only from the simulator's own delivery log. |
| Read future orders or the scorer | The simulator runs in a separate `sim` container. The agent only reaches it through HTTP. |
| Call the scoring endpoint | Needs a per-task random token that only the sim container has. |
| Tamper with the verifier (fake `python3`, write reward.txt) | A collect hook scores inside `sim`. The result reaches a **separate** verifier container as an artifact. |
| Invalid routes | Rejected: unknown or unreleased orders, duplicates, capacity overruns. |

## Baselines (score, seeds 1–3)
| Level | Greedy (myopic) | Reference (`solution/`, rolling-horizon insertion) |
|---|---|---|
| easy | 0.32–0.72 | 0.72–0.92 |
| medium | 0.28–0.45 | 0.76–0.77 |
| hard | 0.11–0.20 | 0.65–0.72 |

## Failure-mode recipes
Each recipe targets one habit we expect frontier agents to show. `scripts/probe_recipes.py` plays the reference and an "LLM-like" proxy for each habit offline (mean of seeds 1–3):

| Recipe | What it punishes | Reference | Proxy with that habit |
|---|---|---|---|
| `nearest_trap` | serving the closest orders first while far, urgent ones spoil | 0.97 | nearest-first 0.68 |
| `short_fuse` | checking in rarely; orders spoil ~30 min after appearing | 0.92 | waits 60 min 0.62 |
| `rush_hour` | not replanning when a surge of orders lands mid-shift | 0.92 | plan once, append 0.38 |
| `waiting_game` | ignoring time windows; nearby orders open hours later | 0.89 | nearest-first 0.37 |
| `heavy_load` | not planning depot reloads into routes | 0.80 | reload only when empty 0.63 |
| `breakdown` | not reacting when a van breaks down mid-shift | 0.92 | plan once, ignore events 0.62 |
| `overbooked` | refusing to triage when demand exceeds the fleet | 0.52 | serve everything 0.24 |

`overbooked` can't reach 1 by design; compare agents against the reference, not against 1.

## Run
```bash
python make_tasks.py                      # tasks/coldchain-{easy,medium,hard}-s1
python make_tasks.py recipes              # one task per failure-mode recipe
python scripts/probe_recipes.py           # offline check of every recipe
harbor run -p tasks/coldchain-medium-s1 -a oracle                       # reference run
harbor run -p tasks/coldchain-medium-s1 -a opencode -m <provider/model> # frontier agent
harbor view ./jobs
```
More seeds: `python make_tasks.py hard 1 2 3`.

Team plan (interactive): open `docs/plan.html` in a browser.
