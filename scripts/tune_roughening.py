"""Tuning grid for Bayes agent roughening on 8 seeds across 80 shots with multiprocessing."""

from __future__ import annotations

import concurrent.futures
import multiprocessing
from pathlib import Path
import sys
import time

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np

from agents.bayes import BayesAgent
from beamsim.env_api import BeamlineEnv, evaluate_env
from beamsim.lattice import build_synthetic_lattice
from beamsim.sim import BeamlineSim


def worker_run_episode(args_tuple: tuple[int, float, int, int]) -> dict:
    """Worker task running a single episode for one seed and roughening value."""
    seed, roughening, budget, n_particles = args_tuple
    t0 = time.perf_counter()

    env = BeamlineEnv(seed=seed, budget=budget, randomization="medium")
    sim = env._sim
    would_be_s_mean = [0.0] * budget

    def on_shot(shot_idx: int, mean_delta: np.ndarray, p_lost: float):
        th_h = np.clip(mean_delta[:10], -env._theta_max, env._theta_max)
        th_v = np.clip(mean_delta[10:], -env._theta_max, env._theta_max)
        eval_res = sim._evaluate(th_h, th_v, add_noise=False)
        would_be_s_mean[shot_idx - 1] = float(eval_res.s_mean)

    agent = BayesAgent(
        n_particles=n_particles,
        chunk_size=n_particles,
        roughening=roughening,
        seed=seed,
    )

    agent.solve(env, on_shot_callback=on_shot)
    elapsed = time.perf_counter() - t0

    score = evaluate_env(env)
    final_shots = score.shots_used

    # Propagate final score to remaining shots if submitted early
    for k in range(final_shots - 1, budget):
        if would_be_s_mean[k] == 0.0 or k >= final_shots:
            would_be_s_mean[k] = score.sf

    return {
        "seed": seed,
        "roughening": roughening,
        "shots_used": final_shots,
        "final_sf": score.sf,
        "solved": score.solved,
        "reward": score.reward,
        "elapsed_seconds": elapsed,
        "would_be_s_mean": would_be_s_mean,
    }


def main():
    start_wall = time.perf_counter()

    seeds_count = 8
    budget = 80
    n_particles = 25_000
    roughening_grid = [0.5, 0.1, 0.02]
    checkpoints = [30, 40, 50, 60, 80]

    num_workers = max(1, multiprocessing.cpu_count() - 1)
    print("=" * 80)
    print(f"BAYES ROUGHENING TUNING GRID (Workers={num_workers}, Particles={n_particles}, Budget={budget})")
    print("=" * 80)

    # 1. Filter first 8 seeds with S0 < 0.4
    lattice = build_synthetic_lattice(theta_max=0.45e-3)
    eval_seeds = []
    seed_candidate = 0
    while len(eval_seeds) < seeds_count:
        sim = BeamlineSim(lattice=lattice, seed=seed_candidate, spread="medium")
        if sim.initial_result.s_mean < 0.4:
            eval_seeds.append(seed_candidate)
        seed_candidate += 1
    print(f"Selected {len(eval_seeds)} seeds with initial S0 < 0.4: {eval_seeds}\n")

    grid_results: dict[float, list[dict]] = {r: [] for r in roughening_grid}

    # Run each roughening value with parallel worker pool
    for r_val in roughening_grid:
        r_t0 = time.perf_counter()
        print(f"--- Evaluating roughening = {r_val} on {seeds_count} seeds ---")
        tasks = [(s, r_val, budget, n_particles) for s in eval_seeds]

        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            outcomes = list(executor.map(worker_run_episode, tasks))

        grid_results[r_val] = outcomes
        r_wall = time.perf_counter() - r_t0

        # Print per-seed outcomes for this roughening value
        for o in outcomes:
            status = "SOLVED" if o["solved"] else "FAILED"
            print(f"  Seed {o['seed']:03d}: {status:<8} | Final Sf = {o['final_sf']:.4f} | Shots = {o['shots_used']:2d} | Time = {o['elapsed_seconds']:.2f}s")
        print(f"Roughening {r_val} completed in {r_wall:.2f}s wall time.\n")

    total_wall = time.perf_counter() - start_wall

    # Print summary solve rates across shot checkpoints
    print("=" * 80)
    print("TUNING RESULTS: SOLVE RATE (%) AT DIFFERENT SHOT HORIZONS")
    print("=" * 80)
    header = f"{'Roughening':<12} | " + " | ".join([f"At {cp:2d} shots" for cp in checkpoints]) + " | Mean Sf (80) | Mean Rew"
    print(header)
    print("-" * len(header))

    for r_val in roughening_grid:
        outcomes = grid_results[r_val]
        rates = []
        for cp in checkpoints:
            # Solved at or before cp shots if max would-be S_mean up to cp >= 0.95
            solved_at_cp = sum(
                1 for o in outcomes
                if any(o["would_be_s_mean"][k] >= 0.95 for k in range(min(cp, len(o["would_be_s_mean"]))))
            )
            rate_pct = (solved_at_cp / seeds_count) * 100.0
            rates.append(f"{rate_pct:5.1f}% ({solved_at_cp}/{seeds_count})")

        mean_sf = np.mean([o["final_sf"] for o in outcomes])
        mean_rew = np.mean([o["reward"] for o in outcomes])

        row = f"{r_val:<12.2f} | " + " | ".join(rates) + f" |    {mean_sf:.4f}   |   {mean_rew:.3f}"
        print(row)

    print("=" * 80)
    print(f"TOTAL WALL TIME FOR ALL RUNS: {total_wall:.2f} seconds ({total_wall / 60:.2f} minutes)")
    print("=" * 80)


if __name__ == "__main__":
    main()
