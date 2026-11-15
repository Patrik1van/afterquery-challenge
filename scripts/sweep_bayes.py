"""Run honest Bayes sweep on seeds 0-9 for roughening in {0.2, 0.5}."""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
import sys
import time

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from agents.bayes import BayesAgent
from beamsim.env_api import BeamlineEnv, evaluate_env


def run_single(args: tuple[int, float]) -> dict:
    seed, roughening = args
    agent_seed = seed + 1_000_003
    t0 = time.perf_counter()

    env = BeamlineEnv(seed=seed, budget=60, randomization="medium")
    agent = BayesAgent(
        n_particles=100_000,
        chunk_size=25_000,
        roughening=roughening,
        seed=agent_seed,
    )
    agent.solve(env)
    elapsed = time.perf_counter() - t0
    score = evaluate_env(env)

    result = {
        "seed": seed,
        "roughening": roughening,
        "reward": float(score.reward),
        "sf": float(score.sf),
        "s0": float(score.s0),
        "progress": float(score.progress),
        "shots_used": int(score.shots_used),
        "solved": bool(score.solved),
        "elapsed_sec": round(elapsed, 2),
    }
    print(f"[Done] Seed {seed} r={roughening}: reward={result['reward']:.4f}, sf={result['sf']:.4f}, shots={result['shots_used']}, time={result['elapsed_sec']}s", flush=True)
    return result


def main():
    seeds = list(range(10))
    roughenings = [0.2, 0.5]

    tasks = [(s, r) for r in roughenings for s in seeds]
    out_file = repo_root / "bayes_sweep_results.json"

    # Use 3 workers for 4 cores/8GB laptop
    n_workers = min(3, multiprocessing.cpu_count())
    print(f"Starting Bayes sweep: {len(tasks)} tasks with {n_workers} workers...", flush=True)

    results = []
    with multiprocessing.Pool(processes=n_workers) as pool:
        for res in pool.imap_unordered(run_single, tasks):
            results.append(res)
            # Save incrementally
            with open(out_file, "w") as f:
                json.dump(results, f, indent=2)

    print("Bayes sweep completed!", flush=True)


if __name__ == "__main__":
    main()
