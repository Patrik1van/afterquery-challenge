"""Local baseline evaluation and loss histogram analysis for TT20 simulator.

Runs:
1. Loss position histogram over 1000 seeds for candidate theta_max values (0.25, 0.35, 0.45, 0.55 mrad).
2. Do-nothing baseline over 50 seeds.
3. Random-search baseline over 50 seeds using ONLY noisy monitor readings.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

# Ensure repository root is on sys.path for direct execution
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np

from beamsim.lattice import build_synthetic_lattice

from beamsim.scoring import compute_reward
from beamsim.sim import BeamlineSim


def evaluate_loss_histogram(
    candidate_thetas: list[float],
    n_seeds: int = 1000,
) -> float:
    """Print loss position histogram across candidate theta_max values and return best choice."""
    print("=" * 72)
    print(f"LOSS POSITION HISTOGRAM OVER {n_seeds} RANDOM SEEDS (theta = 0, initial Delta_theta)")
    print("=" * 72)

    best_theta = candidate_thetas[0]
    best_score = -1.0

    for th_max in candidate_thetas:
        lattice = build_synthetic_lattice(theta_max=th_max)
        loss_counts: Counter[str] = Counter()
        first_monitor_counts: Counter[int] = Counter()
        s0_list = []

        for seed in range(n_seeds):
            sim = BeamlineSim(lattice=lattice, seed=seed, theta_max=th_max, spread="medium")
            init_res = sim.initial_result
            s0_list.append(init_res.s_mean)

            if not init_res.is_lost:
                loss_counts["not_lost"] += 1
            else:
                h_idx = init_res.first_lost_sem_h
                v_idx = init_res.first_lost_sem_v
                earliest = None
                if h_idx is not None and v_idx is not None:
                    earliest = min(h_idx, v_idx)
                elif h_idx is not None:
                    earliest = h_idx
                elif v_idx is not None:
                    earliest = v_idx

                if earliest is not None:
                    first_monitor_counts[earliest] += 1
                else:
                    loss_counts["after_all_sems"] += 1

        print(f"\n--- Candidate theta_max = {th_max * 1e3:.2f} mrad ---")
        print(f"Initial S0: mean = {np.mean(s0_list):.3f}, seeds with S0 < 0.4: {np.mean(np.array(s0_list) < 0.4) * 100:.1f}%")
        print(f"{'Monitor Lost At':<18} | {'Count':<6} | {'Fraction':<8} | Distribution")
        print("-" * 60)

        mon_1_4_count = 0
        for mon_idx in range(6):
            cnt = first_monitor_counts[mon_idx]
            frac = cnt / n_seeds
            bar = "#" * int(frac * 50)
            label = f"Monitor {mon_idx + 1}"
            print(f"{label:<18} | {cnt:<6} | {frac * 100:5.1f}%  | {bar}")
            if mon_idx < 4:
                mon_1_4_count += cnt

        mon_7_plus = sum(cnt for idx, cnt in first_monitor_counts.items() if idx >= 6)
        print(f"{'Monitor 7+':<18} | {mon_7_plus:<6} | {mon_7_plus / n_seeds * 100:5.1f}%  | {'#' * int(mon_7_plus / n_seeds * 50)}")
        not_lost_cnt = loss_counts['not_lost']
        print(f"{'Not Lost':<18} | {not_lost_cnt:<6} | {not_lost_cnt / n_seeds * 100:5.1f}%  | {'#' * int(not_lost_cnt / n_seeds * 50)}")

        mon1_frac = first_monitor_counts[0] / n_seeds
        mon_1_4_frac = mon_1_4_count / n_seeds
        # Select candidate that maximizes loss within monitors 1-4 without collapsing at monitor 1
        if mon_1_4_frac > best_score and mon1_frac < 0.20:
            best_score = mon_1_4_frac
            best_theta = th_max

    print(f"\nSelected theta_max for evaluation: {best_theta * 1e3:.2f} mrad (monitors 1-4 loss: {best_score * 100:.1f}%)")
    return best_theta




def run_do_nothing(
    lattice,
    seeds: list[int],
    budget: int = 40,
) -> dict:
    """Do-nothing baseline: never commands correctors, Sf = S0, reward = 0."""
    rewards = []
    sf_list = []
    s0_list = []
    solved_list = []

    for seed in seeds:
        sim = BeamlineSim(lattice=lattice, seed=seed, spread="medium")
        s0 = sim.initial_result.s_mean
        sf = s0
        score = compute_reward(s0=s0, sf=sf, shots_used=0, budget=budget)
        rewards.append(score.reward)
        sf_list.append(sf)
        s0_list.append(s0)
        solved_list.append(score.solved)

    return {
        "name": "Do-nothing (nop)",
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "sf_mean": float(np.mean(sf_list)),
        "sf_std": float(np.std(sf_list)),
        "s0_mean": float(np.mean(s0_list)),
        "solved_rate": float(np.mean(solved_list)),
        "shots_mean": 0.0,
    }


def run_random_search(
    lattice,
    seeds: list[int],
    budget: int = 40,
) -> dict:
    """Random search: samples noisy readings only, commands best candidate on final shot."""
    rewards = []
    sf_list = []
    s0_list = []
    solved_list = []
    shots_list = []

    for seed in seeds:
        sim = BeamlineSim(lattice=lattice, seed=seed, spread="medium")
        s0 = sim.initial_result.s_mean
        rng_agent = np.random.default_rng(seed + 10000)

        best_score_measured = -1.0
        best_cmd_h = np.zeros(lattice.h.n_corr)
        best_cmd_v = np.zeros(lattice.v.n_corr)

        # Explore budget - 1 shots
        for _ in range(budget - 1):
            cmd_h = rng_agent.uniform(-lattice.theta_max, lattice.theta_max, size=lattice.h.n_corr)
            cmd_v = rng_agent.uniform(-lattice.theta_max, lattice.theta_max, size=lattice.v.n_corr)

            res = sim.set_correctors(cmd_h, cmd_v)

            scores_h = [
                np.sqrt(max(0.0, 1.0 - abs(2.0 * o.ratio - 1.0))) * min(1.0, o.level)
                for o in res.obs_h
            ]
            scores_v = [
                np.sqrt(max(0.0, 1.0 - abs(2.0 * o.ratio - 1.0))) * min(1.0, o.level)
                for o in res.obs_v
            ]
            measured_score = float(np.mean(scores_h + scores_v))

            if measured_score > best_score_measured:
                best_score_measured = measured_score
                best_cmd_h = cmd_h
                best_cmd_v = cmd_v

        # Final shot with the best setting found
        final_res = sim.set_correctors(best_cmd_h, best_cmd_v)
        sf = final_res.s_mean

        score = compute_reward(s0=s0, sf=sf, shots_used=sim.shots_used, budget=budget)
        rewards.append(score.reward)
        sf_list.append(sf)
        s0_list.append(s0)
        solved_list.append(score.solved)
        shots_list.append(sim.shots_used)

    return {
        "name": "Random search",
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "sf_mean": float(np.mean(sf_list)),
        "sf_std": float(np.std(sf_list)),
        "s0_mean": float(np.mean(s0_list)),
        "solved_rate": float(np.mean(solved_list)),
        "shots_mean": float(np.mean(shots_list)),
    }


def main():
    parser = argparse.ArgumentParser(description="Local evaluation and histogram tuning.")
    parser.add_argument("--n-hist-seeds", type=int, default=1000, help="Number of seeds for histogram.")
    parser.add_argument("--n-eval-seeds", type=int, default=50, help="Number of seeds for baseline eval.")
    parser.add_argument("--budget", type=int, default=40, help="Shots budget.")
    args = parser.parse_args()

    candidate_thetas = [0.25e-3, 0.35e-3, 0.45e-3, 0.55e-3]
    chosen_theta = evaluate_loss_histogram(candidate_thetas, n_seeds=args.n_hist_seeds)

    lattice = build_synthetic_lattice(theta_max=chosen_theta)

    eval_seeds = []
    seed = 0
    while len(eval_seeds) < args.n_eval_seeds:
        sim = BeamlineSim(lattice=lattice, seed=seed, spread="medium")
        if sim.initial_result.s_mean < 0.4:
            eval_seeds.append(seed)
        seed += 1

    print("\n" + "=" * 72)
    print(f"BASELINE RUNS OVER {args.n_eval_seeds} SEEDS (S0 < 0.4, Medium Randomization, Budget={args.budget})")
    print("=" * 72)

    results = [
        run_do_nothing(lattice, eval_seeds, budget=args.budget),
        run_random_search(lattice, eval_seeds, budget=args.budget),
    ]

    print(f"{'Baseline':<20} | {'Reward':<15} | {'Sf':<15} | {'Solved %':<10} | {'Shots':<6}")
    print("-" * 72)
    for r in results:
        rew_str = f"{r['reward_mean']:.3f} +/- {r['reward_std']:.3f}"
        sf_str = f"{r['sf_mean']:.3f} +/- {r['sf_std']:.3f}"
        solv_str = f"{r['solved_rate'] * 100:.1f}%"
        shots_str = f"{r['shots_mean']:.0f}"
        print(f"{r['name']:<20} | {rew_str:<15} | {sf_str:<15} | {solv_str:<10} | {shots_str:<6}")


if __name__ == "__main__":
    main()

