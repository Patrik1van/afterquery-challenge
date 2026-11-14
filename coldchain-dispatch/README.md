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

| `nearest_trap_hidden` | **not experimenting**: vans have 4 unlabeled modes with hidden effects on speed and cargo temperature, shuffled per seed | 0.91 (oracle 0.97) | never experiments 0.34–0.58 |
| `short_fuse_hidden` | same hidden modes on the short-fuse shift | 0.97 (upper bound 0.98) | never experiments 0.45–0.61 |
| `short_fuse_drift` | **not re-checking beliefs**: hidden modes + noisy travel times and reefer sensor, and at a hidden time mid-shift every mode's effect is silently reshuffled | 0.84 (upper bound 0.97) | learns once, never re-checks 0.71 |
| `rush_hour_drift` | same drift + noise on the rush-hour shift | 0.94 (upper bound 0.97) | learns once, never re-checks 0.77 |

`overbooked` can't reach 1 by design; compare agents against the reference, not against 1.

**Hidden-rule recipes** turn the shift into a hypothesis-testing problem, like an equation puzzle with unknown operators. The mode labels are fixed, what each one does is hidden and shuffled per seed, and every experiment costs real shift time. The agent writes its inferred rules to `/app/beliefs.json`, and reward = 0.7 × delivery + 0.3 × rules right. The reference rotates modes across vans for up to the first two hours, measures clean driving intervals (distance moved, reefer temperature), gets all 8 labels right, then exploits. The oracle knows the rules from the start, so oracle minus reference is the price of learning.

## Reference runs
`-a oracle` runs `solution/solve.sh`: a **fair reference** with exactly the agent's information. It only sees orders once they are released, and on hidden-mode shifts it learns the modes by probing (each van tries a different mode for 5 minutes; distance along the served path and reefer temperature give speed and cargo effect). Every 5 minutes it re-plans with insertion + local search + random restarts, warm-started from its last plan, and drops orders that no longer pay for themselves. It acts only through the `dispatch` CLI.

`python3 solution/planner.py dispatch --privileged` is an **upper bound**, not a fair competitor: it rebuilds the exact shift from `params.json`, so it also knows future orders and the true hidden rules.

Mean of seeds 1–3 (`python scripts/probe_recipes.py`):

| Recipe | Upper bound (privileged) | **Fair reference** | First simple reference |
|---|---|---|---|
| nearest_trap | 0.969 | 0.969 | 0.968 |
| short_fuse | 0.952 | 0.958 | 0.921 |
| rush_hour | 0.961 | 0.955 | 0.923 |
| waiting_game | 0.945 | 0.945 | 0.888 |
| heavy_load | 0.858 | 0.858 | 0.797 |
| breakdown | 0.948 | 0.941 | 0.920 |
| overbooked | 0.601 | 0.589 | 0.517 |
| nearest_trap_hidden | 0.985 | 0.983 | 0.970 |
| short_fuse_hidden | 0.981 | 0.970 | 0.913 |
| short_fuse_drift | 0.973 | 0.841 | 0.721 |
| rush_hour_drift | 0.969 | 0.939 | 0.761 |

On fixed-rule shifts foresight is worth little (upper bound vs fair ≤ 0.012): what separates good and bad play is decision quality, not information. On **drift** shifts the gap opens (0.03–0.13): even a careful learner pays to notice that the rules changed and to re-learn them under noise. The fair reference keeps measuring the mode it drives; when its last 4 samples disagree with what it learned, it resets and re-probes. Turning that off ("learns once") costs 0.13–0.17.

## First frontier runs (Harbor + OpenCode on Modal)
| Task | Gemini 3.8 Flash | Fair reference | Upper bound | What happened |
|---|---|---|---|---|
| nearest_trap_hidden s1 | 0.982 | 0.986 | 0.988 | Tested all 4 modes with 5-min probes in ~15 min, all rules right, 18/18 on time |
| overbooked s1 (5 runs) | mean 0.602 (0.557–0.640) | 0.655 | 0.656 | Every run wrote its own optimiser (custom, OR-Tools, MIP, ALNS); all 5 below the reference, losing mostly late-arriving orders |

| short_fuse_drift s1 (3 runs) | mean 0.798 (0.763 / 0.784 / 0.847) | 0.900 | — | All 3 below the reference. Beliefs mostly track the end-of-shift rules (0.63–0.88 right vs 0.38–0.63 for the start rules), but its programs misjudge cargo warming under noise: 1–4 orders per run are driven to the customer and arrive spoiled |
| rush_hour_drift s1 (3 runs) | mean 0.855 (0.792 / 0.818 / 0.956) | 0.968 | — | All 3 below the reference. In 2 of 3 runs every lost order falls after the silent rule change (3–4 deliveries arrive spoiled after minute 300); the reference loses none |

In the runs the agent first tried to game the environment (port scans, filesystem search, hidden endpoints like `/reset`, `/score`); none worked.

## Run
```bash
python make_tasks.py                      # tasks/coldchain-{easy,medium,hard}-s1
python make_tasks.py recipes              # one task per failure-mode recipe
python scripts/probe_recipes.py           # offline check of every recipe
harbor run -p tasks/coldchain-medium-s1 -a oracle                       # reference run
harbor run -p tasks/coldchain-medium-s1 -a opencode -m <provider/model> # frontier agent
harbor run -p tasks/coldchain-medium-s1 -a opencode -m <provider/model> -k 5 -n 5 --env modal  # 5 parallel attempts, mean reward
harbor view ./jobs
```
More seeds: `python make_tasks.py hard 1 2 3`.

Team plan (interactive): open `docs/plan.html` in a browser.
