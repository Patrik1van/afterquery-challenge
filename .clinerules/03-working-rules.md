# Working rules

## Reward integrity — never break these
- Hidden state (true offsets Δθ, perturbed model, RNG seeds, score token) lives **only in the
  MCP sidecar container**. Never copy it, the simulator source, or `tests/` into the agent
  (`main`) container image.
- Score is computed by the server from the **true, noise-free** state — never from anything the
  agent reports.
- Budget, clipping to ±θmax, NaN/inf rejection and the frozen-after-submit rule are enforced
  **server-side**.
- The `/score` endpoint requires a token that only the verifier (`tests/`) and the sidecar know;
  calling it freezes the episode.
- The reference agent may only use the same MCP tools the LLM gets.
- Every change to the reward or tools: re-run do-nothing, stop-early, random-search and the
  reference, and check the ordering is still reference ≫ random > stop-early > do-nothing ≈ 0.

## Reproducibility
- Everything seeded (lattice, offsets, perturbations, monitor noise). Same seed + same actions ⇒
  identical readings and score.
- Keep the single eval command (`scripts/run_eval.sh`) working at all times.

## Code
- Python 3.11+, NumPy/SciPy only in the simulator. Type hints, small functions, docstrings that
  cite the report's equation numbers (e.g. `# eq. (2)`).
- Every simulator change comes with a pytest. Run `pytest -q` before saying a task is done.
- Prefer fast local checks (`scripts/local_eval.py`, no Docker) before Harbor runs.
- Don't add dependencies without saying why.

## Secrets & git
- API keys live in `.env` (git-ignored) and are read from environment variables. Never commit
  keys, never print them in logs.
- Work on a feature branch; small commits with clear messages; don't force-push shared branches;
  don't push to `main` directly.
- Before a long or costly step (many Gemini runs, big refactor), state the plan and ask first.
