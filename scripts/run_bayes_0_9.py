"""Run Bayes on seeds 0-9 with budget 60 and medium randomization."""

import concurrent.futures
import json
import multiprocessing
from pathlib import Path
import sys
import time

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np

from agents.bayes import BayesAgent
from beamsim.env_api import BeamlineEnv, evaluate_env, get_env_true_trajectory
from beamsim.lattice import build_synthetic_lattice
from beamsim.sim import BeamlineSim


def worker_bayes(seed: int) -> dict:
    t0 = time.perf_counter()
    budget = 60
    env = BeamlineEnv(seed=seed, budget=budget, randomization="medium")
    agent = BayesAgent(
        n_particles=25_000,
        chunk_size=25_000,
        roughening=0.02,
        seed=seed,
    )
    logs = agent.solve(env)
    elapsed = time.perf_counter() - t0

    score = evaluate_env(env)
    true_traj = get_env_true_trajectory(env)

    # Save episode log to runs/bayes/seed_XXX.json
    out_dir = repo_root / "runs" / "bayes"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"seed_{seed:03d}.json"

    log_data = {
        "metadata": {
            "agent": "bayes",
            "seed": seed,
            "budget": budget,
            "roughening": 0.02,
            "randomization": "medium",
            "elapsed_seconds": round(elapsed, 4),
        },
        "shots": logs,
        "evaluation": {
            **score.to_dict(),
            "true_trajectory": true_traj,
        },
    }
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)

    category = "SOLVED" if score.solved else ("CLOSE_UNDER_095" if score.sf >= 0.80 else ("NEVER_FOUND" if score.sf < 0.50 else "BUDGET_EXHAUSTED"))
    return {
        "seed": seed,
        "solved": score.solved,
        "s0": score.s0,
        "sf": score.sf,
        "shots": score.shots_used,
        "reward": score.reward,
        "category": category,
        "time": elapsed,
        "last_action": logs[-1]["action_type"] if logs else "none",
    }


def main():
    seeds_count = 10
    budget = 60
    lattice = build_synthetic_lattice(theta_max=0.45e-3)
    eval_seeds = []
    seed_cand = 0
    while len(eval_seeds) < seeds_count:
        sim = BeamlineSim(lattice=lattice, seed=seed_cand, spread="medium")
        if sim.initial_result.s_mean < 0.4:
            eval_seeds.append(seed_cand)
        seed_cand += 1

    workers = max(1, multiprocessing.cpu_count() - 1)
    print("=" * 80)
    print(f"RUNNING BAYES (r=0.02, budget=60, medium) ON SEEDS 0-9 (Workers={workers})")
    print("=" * 80)

    t0 = time.perf_counter()
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
        results = list(executor.map(worker_bayes, eval_seeds))
    total_time = time.perf_counter() - t0

    print("\n" + "=" * 80)
    print("PER-SEED RESULTS: BAYES (SEEDS 0-9, BUDGET 60, ROUGHENING 0.02)")
    print("=" * 80)
    print(f"{'Seed':<6} | {'Solved':<8} | {'S0':<8} | {'Sf':<8} | {'Shots':<6} | {'Reward':<8} | {'Category':<18} | {'Time':<6}")
    print("-" * 80)

    for r in results:
        solv = "YES" if r["solved"] else "NO"
        print(f"{r['seed']:<6d} | {solv:<8} | {r['s0']:<8.4f} | {r['sf']:<8.4f} | {r['shots']:<6d} | {r['reward']:<8.4f} | {r['category']:<18} | {r['time']:5.1f}s")

    rewards = [r["reward"] for r in results]
    sfs = [r["sf"] for r in results]
    solved_cnt = sum(1 for r in results if r["solved"])
    shots_list = [r["shots"] for r in results]

    print("=" * 80)
    print(f"SUMMARY (Seeds 0-9):")
    print(f"  Mean Reward : {np.mean(rewards):.4f} +/- {np.std(rewards):.4f}")
    print(f"  Mean Sf     : {np.mean(sfs):.4f} +/- {np.std(sfs):.4f}")
    print(f"  Solve Rate  : {solved_cnt}/{len(results)} ({solved_cnt / len(results) * 100:.1f}%)")
    print(f"  Mean Shots  : {np.mean(shots_list):.1f} +/- {np.std(shots_list):.1f}")
    print(f"  Total Time  : {total_time:.2f}s ({total_time / len(results):.2f}s/seed)")
    print("=" * 80)


if __name__ == "__main__":
    main()
