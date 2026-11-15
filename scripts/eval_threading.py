"""Evaluate ThreadingAgent on seeds 0-9."""

from __future__ import annotations

import json
import multiprocessing
from pathlib import Path
import sys
import time

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from agents.threading import ThreadingAgent
from beamsim.env_api import BeamlineEnv, evaluate_env


def run_single(seed: int) -> dict:
    agent_seed = seed + 1_000_003
    t0 = time.perf_counter()

    env = BeamlineEnv(seed=seed, budget=60, randomization="medium")
    agent = ThreadingAgent(seed=agent_seed)
    agent.solve(env)
    elapsed = time.perf_counter() - t0
    score = evaluate_env(env)

    result = {
        "seed": seed,
        "reward": float(score.reward),
        "sf": float(score.sf),
        "s0": float(score.s0),
        "progress": float(score.progress),
        "shots_used": int(score.shots_used),
        "solved": bool(score.solved),
        "elapsed_sec": round(elapsed, 2),
    }
    print(
        f"[Threading Done] Seed {seed}: reward={result['reward']:.4f}, "
        f"sf={result['sf']:.4f}, s0={result['s0']:.4f}, shots={result['shots_used']}, "
        f"solved={result['solved']}, time={result['elapsed_sec']}s",
        flush=True,
    )
    return result


def main():
    seeds = list(range(10))
    n_workers = min(4, multiprocessing.cpu_count())
    print(f"Starting Threading eval on seeds 0-9 with {n_workers} workers...", flush=True)

    with multiprocessing.Pool(processes=n_workers) as pool:
        results = pool.map(run_single, seeds)

    results.sort(key=lambda d: d["seed"])
    out_file = repo_root / "threading_eval_results.json"
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)

    rewards = [d["reward"] for d in results]
    sfs = [d["sf"] for d in results]
    print("\n=== THREADING EVAL SUMMARY ===")
    for d in results:
        print(f"Seed {d['seed']}: reward={d['reward']:.4f}, sf={d['sf']:.4f}, shots={d['shots_used']}")
    print(f"Mean reward: {sum(rewards)/len(rewards):.4f}")
    print(f"Mean sf:     {sum(sfs)/len(sfs):.4f}")


if __name__ == "__main__":
    main()
