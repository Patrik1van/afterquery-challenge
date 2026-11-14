# TT20 Beam Steering — an RL environment where frontier agents must do real inference

An agent operates a simulated CERN proton transfer line (TT20-like, SPS → North Area).
Every corrector magnet has an unknown misalignment. The agent sets 22 corrector magnets and
sees only noisy split-foil monitor readings. It has to **thread the beam through the whole
900 m line and centre it**, within a limited number of shots.

The physics and the reference algorithm follow Adam Štafa's CERN report *"Algorithms for
Steering the CERN TT20 Transfer Line"* (2026), see `docs/tt20_report.md`.

## Why this is hard (and fair)

- **Partial observability.** Each monitor reports only a foil ratio and an intensity level,
  plus noise. Once the beam hits the pipe, every downstream monitor (in *both* planes) reads
  background only, so the agent is blind beyond the loss point.
- **Inference, not lookup.** The agent gets the full nominal model: response matrices, beam
  widths, apertures, noise level and the physics equations. Success requires combining many
  noisy shots into a belief about 22 unknown offsets and choosing informative next settings.
- **Model mismatch.** The real machine differs from the nominal model (β-beating from initial
  optics mismatch, ±2 % corrector gains, emittance and background spread, after Table 1 of the
  report). The agent is told this.
- **Limited budget.** 60 shots per episode. Random search gets nowhere in 22 dimensions.

Nothing is hidden that a good operator would need: a Bayesian particle filter using
*exactly the same tools* reaches much higher scores (below).

## Environment

| | |
|---|---|
| Goal | Thread and centre the beam: mean intensity-normalised symmetry Ŝ over all 21 monitors (solved at ≥ 0.95) |
| Actions (MCP tools) | `get_machine_info()`, `set_correctors(h[10], v[12])` (1 shot), `get_history()`, `submit()` |
| Observation | per monitor: foil ratio r, level ℓ (noisy, eq. 4 of the report) |
| Hidden state | true offsets Δθ, perturbed optics, gains. These exist only in the `beamline` sidecar container |
| Budget | 60 shots, medium randomisation, θmax = 0.45 mrad, noise ν = 0.25 |

## Reward

```
S0       = true mean Ŝ at the starting setting (θ = 0)
Sf       = true mean Ŝ at the final setting (noise-free, on the real perturbed machine)
progress = clip((Sf − S0) / (0.95 − S0), 0, 1)
reward   = 0.8 · progress + 0.2 · solved · (1 − shots_used / budget)
```

- Doing nothing → exactly 0. Stopping early → close to 0.
- The efficiency bonus applies only to solved episodes, so quitting early can't earn it.
- The score is computed by the simulator from the true state, never from anything the agent reports.

### Reward integrity

- The simulator, the true offsets and the score token live only in the `beamline` sidecar.
  The agent container has no simulator code.
- `/score` requires a per-task token that only the verifier holds (tests are copied in after
  the agent finishes). Calling it freezes the episode.
- The budget, clipping to ±θmax, NaN/Inf rejection and freeze-after-submit are all enforced
  server-side.
- Exploit attempts we tested: TODO (list + result).

## Results

Medium randomisation, budget 60, tasks `beam-steering-s000 … s004`, run with Harbor on Modal.

| Agent | Mean reward | Mean Sf | Notes |
|---|---|---|---|
| Reference: Bayes particle filter (`oracle`) | TODO | TODO | threads the full line on ~8/10 seeds |
| Gemini 3.8 Flash via OpenCode (3 runs) | TODO | TODO | |
| Stop-early (2 shots, submit) | TODO (s000: 0.012) | | |
| Do-nothing (`nop`) | TODO (0.000 expected) | | |
| Random search (local) | 0.020 | 0.025 | |

### Where the frontier agent fails

TODO from transcripts. Pattern seen in pilot runs: Gemini uses the shell to invert the
response matrix and threads the first 4–6 monitors (Sf ≈ 0.28–0.36). It then makes local
one-at-a-time corrector tweaks and doesn't build a belief across shots, so it runs out of
budget before threading the line. The particle filter updates all 22 offsets jointly from
every shot, including "beam lost" observations, which carry information too.

Known limitation of the reference: it trusts the nominal model, so model mismatch limits final
centring to Sf ≈ 0.8. It rarely reaches the 0.95 "solved" threshold, which leaves headroom
above the reference as well.

## Reproduce

Requirements: Python 3.11+, a [Modal](https://modal.com) account (`modal setup`), and a
Gemini API key.

```bash
pip install -r requirements.txt harbor modal
export GEMINI_API_KEY=...            # Windows: $env:GEMINI_API_KEY="..."
./scripts/run_eval.sh                # Windows: scripts\run_eval.ps1
```

This runs the oracle, 3× OpenCode + Gemini, nop and stop-early over all tasks, then prints the
results table. Single runs:

```bash
harbor run -p tasks/beam-steering-s000 -a oracle -e modal
harbor run -p tasks/beam-steering-s000 -a opencode -m gemini/gemini-3.8-flash -e modal
```

The tasks are standard Harbor multi-container tasks and also run locally with `-e docker`.
Fast local checks without Docker: `pytest -q`, `python scripts/local_eval.py`.

## Repository layout

```
beamsim/        simulator (lattice, physics eq. 1–7, randomisation, scoring, agent-facing env_api)
agents/         bayes.py (reference particle filter), baselines.py (random, stop-early)
scripts/        gen_tasks.py, local_eval.py, pilot_llm.py, run_eval.sh/.ps1
tasks/          generated Harbor tasks beam-steering-sNNN
docs/           tt20_report.md (physics source), PLAN.md
```

## Simplifications vs. the real TT20

A synthetic TT20-like lattice is used because the MAD-X optics weren't available. Other
simplifications: Gaussian beam, rectangular apertures, binary loss, uniform noise (as in the
report), and no quadrupole gradient errors.

## Credits

Physics model and Bayesian reference after Adam Štafa (CERN, supervisors A. Menor de Oñate,
M. Schenk). Built for the AfterQuery track at Junction 2026.