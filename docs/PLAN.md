# Beam-Steering RL Environment — Technical Plan

Junction 2026 · AfterQuery challenge · Harbor + OpenCode

Goal: a Harbor task in which an agent steers a simulated TT20-like proton transfer line
(horizontal + vertical plane) through MCP tools, scored on true beam centring. Reference run =
Bayesian particle-filter agent (after Adam Štafa's report). Frontier run = OpenCode + Gemini.

---

## 0. Repo setup (do first)

```bash
git clone https://github.com/Patrik1van/afterquery-challenge && cd afterquery-challenge
git checkout -b dev            # integration branch; keep main = last working demo
# per person: git checkout -b feat/<sim|mcp|bayes|verifier|pilot> off dev, PR into dev
python -m venv .venv && source .venv/bin/activate
pip install numpy scipy fastmcp mcp pytest google-genai
uv tool install harbor        # or: pip install harbor   (check harbor README)
```

Docker must be running locally. **Multi-container tasks (our MCP sidecar) only work with the
`docker` environment provider, not Modal/Daytona** — all evals run on a laptop with Docker.

## 1. Repo layout

```
afterquery-challenge/
  beamsim/                      # pure numpy, no JAX
    lattice.py                  # synthetic TT20-like lattice (both planes)
    sim.py                      # BeamlineSim: trajectory, loss, SEM readings, symmetry
    randomize.py                # model mismatch (corrector gain, beta, emittance, background)
    scoring.py                  # reward function (shared by sidecar + local eval)
  agents/
    bayes.py                    # particle-filter reference agent
    baselines.py                # do-nothing, stop-early, random search, naive pseudo-inverse
  mcp_server/server.py          # FastMCP wrapper around BeamlineSim (copied into each task)
  scripts/
    gen_tasks.py                # writes tasks/beam-steering-sNNN/ from a template + seed
    local_eval.py               # runs all agents vs. sim directly (fast, no Docker)
    pilot_llm.py                # quick Gemini function-calling loop (gap sanity check)
  tasks/beam-steering-s001/ ... # generated Harbor tasks
  tests/                        # pytest for beamsim
  README.md
```

## 2. Simulator (`beamsim/`) — build from Adam's report, section 1.1

**Lattice (`lattice.py`).** Adam's MAD-X Twiss table isn't available → generate a synthetic
TT20-like lattice, deterministic (fixed seed), and say so in the README.
Ask Adam if he can share just the Twiss table (s, βx, βy, μx, μy, aperture) — if yes, drop it in
as `lattice_tt20.csv` and everything else stays the same.
- ~900 m line, FODO-ish: β oscillating ~15–100 m, ~90° phase advance per cell (μ in units of 2π).
- Horizontal: 10 correctors, 11 SEMs. Vertical: 12 correctors, 10 SEMs. Interleave so each
  monitor sees several upstream correctors; make the last 1–2 correctors see only 1 monitor
  (reproduces the near-null directions from the report).
- Aperture half-widths a_h, a_v per element (~20–40 mm), plus a dense set of aperture
  check points (e.g. every few metres) for loss detection.
- Emittance: εn = 2 µm, βγ = 427.35 → ε = εn/βγ; σ_i = sqrt(β_i ε).
- Foil length L per SEM (start with L = 3σ_i; tune).

**Physics (`sim.py`), per plane independently:**
- R_ij = sqrt(β_i β_j) sin(2π(μ_i − μ_j)) if μ_i > μ_j else 0 (eq. 1), for all aperture points
  and monitors.
- Hidden offsets Δθ ~ U[−θmax, θmax]^n. Agent commands θ. Trajectory x = R(Δθ − θ) (eq. 10).
- Loss: first aperture point with |x| > a → beam lost; all monitors downstream read background only.
- Foils (eq. 2): I1 = Φ(−x/σ) − Φ((−L−x)/σ), I2 = Φ((L−x)/σ) − Φ(−x/σ).
- Noise (eq. 3): each foil += U[0, ν], ν = 0.25 default, seeded RNG.
- Observation per SEM (eq. 4/11): ratio r = Ĩ1/(Ĩ1+Ĩ2), level ℓ = (Ĩ1+Ĩ2)/Imax.
- Truth (noise-free) for scoring: S = sqrt(1 − |I1−I2|/(I1+I2)), Ŝ = S·(I1+I2)/Imax (eq. 5–6);
  Ŝ = 0 at lost monitors. `S_mean` = mean Ŝ over all 21 monitors. Solved ⇔ S_mean ≥ 0.95.
- Tune θmax so a random start typically loses the beam within the first ~3 monitors
  (print a histogram of loss positions over 1000 seeds).

**Randomization (`randomize.py`).** Table 1 "medium" spreads: corrector gain ±2 %, βx₀ ±50 %,
βy₀ ±10 %, emittance ±20 %, background ±40 % (log-normal). The agent and the Bayes reference
both get the **nominal** model; the episode runs on the perturbed one. This is a difficulty knob.

**Unit tests:** centred beam → S_mean ≈ 1; θ = Δθ → solved; huge θ → beam lost at monitor 1;
same seed → identical readings.

## 3. MCP server (sidecar container)

State is global in the server process (not per MCP session — OpenCode may reconnect).
Episode config (seed, budget, ν, randomization) comes from env vars baked into the sidecar image.

Tools:
| tool | effect |
|---|---|
| `get_machine_info()` | nominal lattice table, nominal response matrices (monitors × correctors, both planes), σ per monitor, foil length, apertures, ν, θmax, budget, physics description |
| `set_correctors(h: list[10], v: list[12])` | sets absolute angles (clipped to ±θmax, NaN rejected), costs 1 shot, returns per-SEM `{ratio, level}` for both planes + shots remaining |
| `get_history()` | all past settings + readings (fair memory aid) |
| `submit()` | freezes the episode; further `set_correctors` rejected |

Budget enforced server-side (default **40 shots**; Bayes needs ~30). After budget is exhausted the
last setting stands. The final state = last commanded setting, submitted or not.

Non-MCP endpoint `GET /score?token=…` returns the reward JSON — token is a random string baked into
the sidecar image env **and** into `tests/` (which Harbor copies in only after the agent finishes),
so the agent can't use it as an oracle. Calling `/score` also freezes the episode.

## 4. Reward (`scoring.py`)

```
S0      = S_mean at the initial setting θ = 0            (task generator keeps only seeds with S0 < 0.4)
Sf      = S_mean at the final setting (noise-free, true perturbed model)
progress = clip((Sf − S0) / (0.95 − S0), 0, 1)
solved   = Sf ≥ 0.95
reward   = 0.8·progress + 0.2·solved·(1 − shots_used / budget)
```
- Do-nothing → 0. Stop-early after a few shots → small. Efficiency bonus only if solved.
- Making things worse can't go below 0; random search is capped by the budget.
- Verifier writes this single number to `/logs/verifier/reward.txt`; also dump the full JSON
  (Sf, S0, shots, per-monitor Ŝ, lost-at) to `/logs/verifier/details.json` for analysis.

## 5. Harbor task (copy `harbor/examples/tasks/hello-mcp` as the template)

```
tasks/beam-steering-s001/
  task.toml               # [[environment.mcp_servers]] name="beamline", streamable-http,
                          #   url="http://beamline:8000/mcp"; agent timeout ~1800 s
  instruction.md          # the problem in operator language: goal, tools, budget, scoring rule,
                          #   that the model is nominal & may be slightly off. No tricks.
  environment/
    Dockerfile            # agent container: python3 + numpy (agent may write its own code)
    docker-compose.yaml   # adds `beamline` service with healthcheck (as in hello-mcp)
    beamline/Dockerfile   # python:3.12-slim + numpy scipy fastmcp; COPY beamsim + server.py
    beamline/server.py
  tests/test.sh           # curl http://beamline:8000/score?token=… → reward.txt
  solution/solve.sh       # runs agents/bayes.py as an MCP client (only uses the tools!)
```
Do NOT copy `beamline/` into the agent container. Generate N tasks (e.g. 10 seeds; maybe two
difficulty tiers) with `scripts/gen_tasks.py`.

## 6. Reference agent (`agents/bayes.py`) — report section 3

- Two independent particle filters (H: 10-dim, V: 12-dim), uniform prior on [−θmax, θmax].
- Particles: start at 2·10⁴ per plane (Adam used 10⁵ total; reduce if slow).
- Approximate likelihood (eq. 14–15): residuals of ratio and level, normalized by ν/T̃ and 2ν/Imax,
  Huber loss; drop ratio term where particle predicts beam lost.
- Multinomial resampling + roughening with 0.5·Σ^{1/2}ε (eq. 16).
- Action (eq. 17): θ = mean + (0.1 + 0.6·p_lost)(sample − mean).
- Stop and `submit()` when the predicted S_mean under the posterior mean ≥ 0.97, or budget out.
- Must talk to the env **only via MCP tools**, so it's an honest reference.

## 7. Baselines (`agents/baselines.py`)

do-nothing (`-a nop`), stop-early (1–3 shots then submit), random search, naive operator
(estimate positions from ratios on visible monitors → pseudo-inverse). Report all in README.

## 8. Order of work (time-boxed)

1. **Sim + scoring + unit tests** (~3 h). `local_eval.py` with do-nothing & random baselines.
2. **Bayes agent against sim directly** (~3 h). Target: ≥90 % solved within 40 shots.
3. **Pilot gap check** (`pilot_llm.py`, ~1 h): Gemini function-calling loop, same tools as MCP,
   plus a `run_python` tool so it isn't handicapped vs. OpenCode. 3–5 episodes.
   - If Gemini solves easily → turn knobs: lower budget, medium→high randomization, raise ν.
   - If it can't even improve → check instructions/tools are clear (fairness criterion).
4. **Wrap in Harbor** (~3 h): MCP server, compose, verifier, solution; `harbor run -a oracle`.
5. **Real runs**: `-a nop`, stop-early, `-a opencode -m <gemini model>` × 3 per task.
6. **Red-team the reward** (1 h): try to read sidecar, call /score, spam shots, NaNs, never submit.
7. **README + demo** (trajectory plot per step, Bayes belief plot, results table).
8. *Stretch:* SFT a small open model (Qwen ~1.5–3B) on Modal using successful trajectories;
   compare before/after on held-out seeds.

## 9. Commands (fill in exact model string)

```bash
export GEMINI_API_KEY=...   # check which env var Harbor's opencode agent expects for Google
harbor run -p tasks/beam-steering-s001 -a oracle                 # reference (Bayes)
harbor run -p tasks/beam-steering-s001 -a nop                    # do-nothing
harbor run -p tasks/beam-steering-s001 -a opencode -m google/<gemini-model> -k 3
```
Wrap these in `scripts/run_eval.sh` → one command for judges, prints a results table.

## Open questions

- Can Adam share the TT20 Twiss table / give a sanity check of our lattice?
- Who owns which branch (sim / mcp+harbor / bayes / pilot+README)?
