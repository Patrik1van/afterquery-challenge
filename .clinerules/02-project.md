# Our project: TT20 beam steering environment

The agent steers a simulated CERN TT20-like proton transfer line by setting corrector dipole
magnets, observing only noisy split-foil monitor (SEM) readings. Goal: thread and centre the
beam through all monitors within a limited number of shots.

## Source material (read before working on physics or the reference agent)
- `docs/tt20_report.md` — Adam Štafa's report, converted to Markdown. **Section 1 = the
  simulator spec** (equations 1–7). **Section 3 = the Bayesian reference agent** (eq. 10–17).
- `docs/papers/` — original PDFs (report + slides), for humans.
- `docs/PLAN.md` — the full technical plan: layout, tools, reward, order of work. Follow it.
  If you need to deviate, say so and update PLAN.md in the same change.

## Fixed design decisions
- **Both planes** (horizontal: 10 correctors, 11 SEMs; vertical: 12 correctors, 10 SEMs),
  planes are independent (uncoupled linear optics).
- We **build the simulator ourselves in NumPy** (no JAX). Adam's MAD-X lattice is not available,
  so we use a deterministic **synthetic TT20-like lattice**; must be swappable for a real Twiss
  table later (`lattice_tt20.csv`).
- Hidden per-corrector offsets Δθ ~ U[−θmax, θmax]; trajectory x = R(Δθ − θ).
- Model mismatch ("medium" randomization from the report's Table 1) is a difficulty knob: the
  agent and the reference get the **nominal** model, the episode runs on the perturbed one.
- MCP tools: `get_machine_info`, `set_correctors(h[10], v[12])` (costs 1 shot), `get_history`,
  `submit`. Default budget 40 shots, noise ν = 0.25.
- Reward: `0.8·progress + 0.2·solved·(1 − shots_used/budget)`, progress measured from the
  starting symmetry, so do-nothing = 0. Solved ⇔ mean intensity-normalized symmetry ≥ 0.95.
- Reference run = Bayesian particle filter (Harbor `oracle` via `solution/solve.sh`), using
  **only the MCP tools** — never the hidden state.
- Frontier run = OpenCode + Gemini. Evals run locally with Docker (multi-container tasks don't
  work on Modal).
