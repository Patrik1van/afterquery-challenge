"""Local baseline evaluation, loss histogram analysis, and Bayes reference run for TT20 simulator.

Runs:
1. Loss position histogram over 1000 seeds for candidate theta_max values.
2. Do-nothing baseline over 50 seeds (saves runs/nop/seed_XXX.json).
3. Random-search baseline over 50 seeds (saves runs/random/seed_XXX.json).
4. Bayes particle-filter agent over 50 seeds (saves runs/bayes/seed_XXX.json).
5. Prints summary table and per-seed failure diagnostic table.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
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
        if mon_1_4_frac > best_score and mon1_frac < 0.20:
            best_score = mon_1_4_frac
            best_theta = th_max

    print(f"\nSelected theta_max for evaluation: {best_theta * 1e3:.2f} mrad (monitors 1-4 loss: {best_score * 100:.1f}%)")
    return best_theta





def save_episode_log(
    agent_name: str,
    seed: int,
    budget: int,
    elapsed_sec: float,
    shots_history: list[dict[str, Any]],
    score_details: dict[str, Any],
    true_traj: dict[str, Any],
) -> Path:
    """Save replay episode log to runs/<agent>/seed_XXX.json."""
    out_dir = repo_root / "runs" / agent_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"seed_{seed:03d}.json"

    log_data = {
        "metadata": {
            "agent": agent_name,
            "seed": seed,
            "budget": budget,
            "randomization": "medium",
            "elapsed_seconds": round(elapsed_sec, 4),
        },
        "shots": shots_history,
        "evaluation": {
            **score_details,
            "true_trajectory": true_traj,
        },
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(log_data, f, indent=2)
    return out_file


def run_do_nothing(
    lattice,
    seeds: list[int],
    budget: int = 40,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Do-nothing baseline: never commands correctors, Sf = S0, reward = 0."""
    rewards = []
    sf_list = []
    s0_list = []
    solved_list = []
    times = []
    per_seed = []

    for seed in seeds:
        out_file = repo_root / "runs" / "nop" / f"seed_{seed:03d}.json"
        if out_file.exists():
            try:
                with open(out_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                sc = cached["evaluation"]
                rewards.append(sc["reward"])
                sf_list.append(sc["sf"])
                s0_list.append(sc["s0"])
                solved_list.append(sc["solved"])
                times.append(cached["metadata"].get("elapsed_seconds", 0.0))
                per_seed.append({
                    "seed": seed,
                    "solved": sc["solved"],
                    "s0": sc["s0"],
                    "sf": sc["sf"],
                    "shots": 0,
                    "reward": sc["reward"],
                    "category": "SOLVED" if sc["solved"] else "NEVER_FOUND",
                })
                continue
            except Exception:
                pass

        t0 = time.perf_counter()
        env = BeamlineEnv(seed=seed, budget=budget, randomization="medium", lattice=lattice)
        env.submit()
        elapsed = time.perf_counter() - t0

        score = evaluate_env(env)
        true_traj = get_env_true_trajectory(env)

        save_episode_log("nop", seed, budget, elapsed, [], score.to_dict(), true_traj)

        rewards.append(score.reward)
        sf_list.append(score.sf)
        s0_list.append(score.s0)
        solved_list.append(score.solved)
        times.append(elapsed)

        per_seed.append({
            "seed": seed,
            "solved": score.solved,
            "s0": score.s0,
            "sf": score.sf,
            "shots": 0,
            "reward": score.reward,
            "category": "SOLVED" if score.solved else "NEVER_FOUND",
        })

    return {
        "name": "Do-nothing (nop)",
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "sf_mean": float(np.mean(sf_list)),
        "sf_std": float(np.std(sf_list)),
        "s0_mean": float(np.mean(s0_list)),
        "solved_rate": float(np.mean(solved_list)),
        "shots_mean": 0.0,
        "shots_std": 0.0,
        "time_per_ep": float(np.mean(times)),
    }, per_seed




def run_random_search(
    lattice,
    seeds: list[int],
    budget: int = 40,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Random search: samples noisy readings only, commands best candidate on final shot."""
    rewards = []
    sf_list = []
    s0_list = []
    solved_list = []
    shots_list = []
    times = []
    per_seed = []

    for seed in seeds:
        out_file = repo_root / "runs" / "random" / f"seed_{seed:03d}.json"
        if out_file.exists():
            try:
                with open(out_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                sc = cached["evaluation"]
                rewards.append(sc["reward"])
                sf_list.append(sc["sf"])
                s0_list.append(sc["s0"])
                solved_list.append(sc["solved"])
                shots_list.append(sc["shots_used"])
                times.append(cached["metadata"].get("elapsed_seconds", 0.0))
                category = "SOLVED" if sc["solved"] else ("CLOSE_UNDER_095" if sc["sf"] >= 0.80 else ("NEVER_FOUND" if sc["sf"] < 0.50 else "BUDGET_EXHAUSTED"))
                per_seed.append({
                    "seed": seed,
                    "solved": sc["solved"],
                    "s0": sc["s0"],
                    "sf": sc["sf"],
                    "shots": sc["shots_used"],
                    "reward": sc["reward"],
                    "category": category,
                })
                continue
            except Exception:
                pass

        t0 = time.perf_counter()

        env = BeamlineEnv(seed=seed, budget=budget, randomization="medium", lattice=lattice)
        rng_agent = np.random.default_rng(seed + 10000)

        best_score_measured = -1.0
        best_cmd_h = np.zeros(lattice.h.n_corr)
        best_cmd_v = np.zeros(lattice.v.n_corr)
        shots_history = []

        # Explore budget - 1 shots
        for _ in range(budget - 1):
            cmd_h = rng_agent.uniform(-lattice.theta_max, lattice.theta_max, size=lattice.h.n_corr)
            cmd_v = rng_agent.uniform(-lattice.theta_max, lattice.theta_max, size=lattice.v.n_corr)

            obs = env.set_correctors(cmd_h, cmd_v)

            scores_h = [
                np.sqrt(max(0.0, 1.0 - abs(2.0 * o["ratio"] - 1.0))) * min(1.0, o["level"])
                for o in obs["h"]
            ]
            scores_v = [
                np.sqrt(max(0.0, 1.0 - abs(2.0 * o["ratio"] - 1.0))) * min(1.0, o["level"])
                for o in obs["v"]
            ]
            measured_score = float(np.mean(scores_h + scores_v))

            shots_history.append({
                "shot": obs["shots_used"],
                "theta_h": cmd_h.tolist(),
                "theta_v": cmd_v.tolist(),
                "obs_h": obs["h"],
                "obs_v": obs["v"],
                "posterior_mean_h": None,
                "posterior_mean_v": None,
                "p_lost": None,
                "predicted_s_mean": None,
                "action_type": "random_explore",
            })

            if measured_score > best_score_measured:
                best_score_measured = measured_score
                best_cmd_h = cmd_h
                best_cmd_v = cmd_v

        # Final shot with the best setting found
        final_obs = env.set_correctors(best_cmd_h, best_cmd_v)
        shots_history.append({
            "shot": final_obs["shots_used"],
            "theta_h": best_cmd_h.tolist(),
            "theta_v": best_cmd_v.tolist(),
            "obs_h": final_obs["h"],
            "obs_v": final_obs["v"],
            "posterior_mean_h": None,
            "posterior_mean_v": None,
            "p_lost": None,
            "predicted_s_mean": None,
            "action_type": "final_best",
        })
        env.submit()
        elapsed = time.perf_counter() - t0

        score = evaluate_env(env)
        true_traj = get_env_true_trajectory(env)

        save_episode_log("random", seed, budget, elapsed, shots_history, score.to_dict(), true_traj)

        rewards.append(score.reward)
        sf_list.append(score.sf)
        s0_list.append(score.s0)
        solved_list.append(score.solved)
        shots_list.append(score.shots_used)
        times.append(elapsed)

        category = "SOLVED" if score.solved else ("CLOSE_UNDER_095" if score.sf >= 0.80 else ("NEVER_FOUND" if score.sf < 0.50 else "BUDGET_EXHAUSTED"))
        per_seed.append({
            "seed": seed,
            "solved": score.solved,
            "s0": score.s0,
            "sf": score.sf,
            "shots": score.shots_used,
            "reward": score.reward,
            "category": category,
        })

    return {
        "name": "Random search",
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "sf_mean": float(np.mean(sf_list)),
        "sf_std": float(np.std(sf_list)),
        "s0_mean": float(np.mean(s0_list)),
        "solved_rate": float(np.mean(solved_list)),
        "shots_mean": float(np.mean(shots_list)),
        "shots_std": float(np.std(shots_list)),
        "time_per_ep": float(np.mean(times)),
    }, per_seed


def _bayes_seed_worker(args_tuple: tuple[int, int, int, int, float]) -> dict[str, Any]:
    """Top-level worker function for multiprocessing Bayes agent runs."""
    seed, budget, n_particles, chunk_size, roughening = args_tuple
    t0 = time.perf_counter()
    env = BeamlineEnv(seed=seed, budget=budget, randomization="medium")
    agent = BayesAgent(n_particles=n_particles, chunk_size=chunk_size, roughening=roughening, seed=seed)
    agent_logs = agent.solve(env)
    elapsed = time.perf_counter() - t0
    score = evaluate_env(env)
    true_traj = get_env_true_trajectory(env)
    save_episode_log("bayes", seed, budget, elapsed, agent_logs, score.to_dict(), true_traj)
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
    }


def run_bayes(
    lattice,
    seeds: list[int],
    budget: int = 40,
    n_particles: int = 50_000,
    chunk_size: int = 10_000,
    roughening: float = 0.5,
    workers: int = 1,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Bayes reference agent: runs joint 22D particle filter, saves runs/bayes/seed_XXX.json."""
    per_seed: dict[int, dict[str, Any]] = {}
    uncached_seeds: list[int] = []

    print(f"\nRunning Bayes Agent on {len(seeds)} seeds ({n_particles} particles, roughening={roughening}, workers={workers})...")
    for idx, seed in enumerate(seeds, 1):
        out_file = repo_root / "runs" / "bayes" / f"seed_{seed:03d}.json"
        if out_file.exists():
            try:
                with open(out_file, "r", encoding="utf-8") as f:
                    cached = json.load(f)
                sc = cached["evaluation"]
                category = "SOLVED" if sc["solved"] else ("CLOSE_UNDER_095" if sc["sf"] >= 0.80 else ("NEVER_FOUND" if sc["sf"] < 0.50 else "BUDGET_EXHAUSTED"))
                per_seed[seed] = {
                    "seed": seed,
                    "solved": sc["solved"],
                    "s0": sc["s0"],
                    "sf": sc["sf"],
                    "shots": sc["shots_used"],
                    "reward": sc["reward"],
                    "category": category,
                    "time": cached["metadata"].get("elapsed_seconds", 0.0),
                }
                print(f"  [{idx:02d}/{len(seeds):02d}] Seed {seed:03d}: [CACHED] {category:<17} | Sf = {sc['sf']:.4f} | Shots = {sc['shots_used']:2d} | Rew = {sc['reward']:.3f}")
                continue
            except Exception:
                pass
        uncached_seeds.append(seed)

    if uncached_seeds:
        tasks = [(s, budget, n_particles, chunk_size, roughening) for s in uncached_seeds]
        if workers > 1 and len(tasks) > 1:
            with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as executor:
                results = list(executor.map(_bayes_seed_worker, tasks))
            for res in results:
                per_seed[res["seed"]] = res
                status = "SOLVED" if res["solved"] else f"FAILED ({res['category']})"
                print(f"  Seed {res['seed']:03d}: {status:<24} | Sf = {res['sf']:.4f} | Shots = {res['shots']:2d} | Rew = {res['reward']:.3f} | {res['time']:.2f}s")
        else:
            for task in tasks:
                res = _bayes_seed_worker(task)
                per_seed[res["seed"]] = res
                status = "SOLVED" if res["solved"] else f"FAILED ({res['category']})"
                print(f"  Seed {res['seed']:03d}: {status:<24} | Sf = {res['sf']:.4f} | Shots = {res['shots']:2d} | Rew = {res['reward']:.3f} | {res['time']:.2f}s")

    ordered_per_seed = [per_seed[s] for s in seeds]
    rewards = [p["reward"] for p in ordered_per_seed]
    sf_list = [p["sf"] for p in ordered_per_seed]
    s0_list = [p["s0"] for p in ordered_per_seed]
    solved_list = [p["solved"] for p in ordered_per_seed]
    shots_list = [p["shots"] for p in ordered_per_seed]
    times = [p["time"] for p in ordered_per_seed]

    return {
        "name": "Bayes (reference)",
        "reward_mean": float(np.mean(rewards)),
        "reward_std": float(np.std(rewards)),
        "sf_mean": float(np.mean(sf_list)),
        "sf_std": float(np.std(sf_list)),
        "s0_mean": float(np.mean(s0_list)),
        "solved_rate": float(np.mean(solved_list)),
        "shots_mean": float(np.mean(shots_list)),
        "shots_std": float(np.std(shots_list)),
        "time_per_ep": float(np.mean(times)),
    }, ordered_per_seed





def main():
    parser = argparse.ArgumentParser(description="Local evaluation, loss histogram, and Bayes reference run.")
    parser.add_argument("--n-hist-seeds", type=int, default=1000, help="Number of seeds for histogram.")
    parser.add_argument("--n-eval-seeds", type=int, default=50, help="Number of seeds for baseline eval.")
    parser.add_argument("--budget", type=int, default=40, help="Shots budget.")
    parser.add_argument("--n-particles", type=int, default=50_000, help="Particle count for Bayes agent.")
    parser.add_argument("--chunk-size", type=int, default=25_000, help="Chunk size for particles.")
    parser.add_argument("--roughening", type=float, default=0.5, help="Roughening jitter coefficient.")
    parser.add_argument("--workers", type=int, default=max(1, multiprocessing.cpu_count() - 1), help="Number of parallel workers.")
    parser.add_argument("--skip-hist", action="store_true", help="Skip candidate histogram tuning and use default theta_max.")
    args = parser.parse_args()

    if not args.skip_hist:
        candidate_thetas = [0.25e-3, 0.35e-3, 0.45e-3, 0.55e-3]
        chosen_theta = evaluate_loss_histogram(candidate_thetas, n_seeds=args.n_hist_seeds)
    else:
        chosen_theta = 0.45e-3

    lattice = build_synthetic_lattice(theta_max=chosen_theta)

    eval_seeds = []
    seed = 0
    while len(eval_seeds) < args.n_eval_seeds:
        sim = BeamlineSim(lattice=lattice, seed=seed, spread="medium")
        if sim.initial_result.s_mean < 0.4:
            eval_seeds.append(seed)
        seed += 1

    print("\n" + "=" * 80)
    print(f"EVALUATION OVER {args.n_eval_seeds} SEEDS (S0 < 0.4, Medium Randomization, Budget={args.budget}, Workers={args.workers})")
    print("=" * 80)

    nop_summary, nop_per_seed = run_do_nothing(lattice, eval_seeds, budget=args.budget)
    rnd_summary, rnd_per_seed = run_random_search(lattice, eval_seeds, budget=args.budget)
    bayes_summary, bayes_per_seed = run_bayes(
        lattice,
        eval_seeds,
        budget=args.budget,
        n_particles=args.n_particles,
        chunk_size=args.chunk_size,
        roughening=args.roughening,
        workers=args.workers,
    )



    summaries = [nop_summary, rnd_summary, bayes_summary]

    print("\n" + "=" * 80)
    print("SUMMARY RESULTS (50 SEEDS)")
    print("=" * 80)
    print(f"{'Agent / Baseline':<20} | {'Reward':<17} | {'Sf':<17} | {'Solved %':<10} | {'Shots':<12} | {'Sec / Ep':<8}")
    print("-" * 92)
    for s in summaries:
        rew_str = f"{s['reward_mean']:.3f} +/- {s['reward_std']:.3f}"
        sf_str = f"{s['sf_mean']:.3f} +/- {s['sf_std']:.3f}"
        solv_str = f"{s['solved_rate'] * 100:5.1f}%"
        shots_str = f"{s['shots_mean']:4.1f} +/- {s['shots_std']:4.1f}"
        sec_str = f"{s['time_per_ep']:5.2f}s"
        print(f"{s['name']:<20} | {rew_str:<17} | {sf_str:<17} | {solv_str:<10} | {shots_str:<12} | {sec_str:<8}")

    print("\n" + "=" * 80)
    print("PER-SEED BREAKDOWN FOR BAYES AGENT")
    print("=" * 80)
    print(f"{'Seed':<6} | {'Solved':<8} | {'S0':<8} | {'Sf':<8} | {'Shots':<6} | {'Reward':<8} | {'Category':<18}")
    print("-" * 80)
    for p in bayes_per_seed:
        solv_flag = "YES" if p["solved"] else "NO"
        print(f"{p['seed']:<6d} | {solv_flag:<8} | {p['s0']:<8.4f} | {p['sf']:<8.4f} | {p['shots']:<6d} | {p['reward']:<8.4f} | {p['category']:<18}")

    failures = [p for p in bayes_per_seed if not p["solved"]]
    if failures:
        print("\n" + "=" * 80)
        print(f"BAYES FAILURE DIAGNOSTICS ({len(failures)} / {len(bayes_per_seed)} seeds)")
        print("=" * 80)
        fail_counts = Counter(p["category"] for p in failures)
        for cat, cnt in fail_counts.items():
            print(f"  - {cat}: {cnt} seeds ({cnt / len(bayes_per_seed) * 100:.1f}%)")
        for f_item in failures:
            print(f"    Seed {f_item['seed']:03d}: S0={f_item['s0']:.4f}, Sf={f_item['sf']:.4f}, shots={f_item['shots']}, category={f_item['category']}")
    else:
        print("\n" + "=" * 80)
        print("BAYES AGENT SOLVED 100% OF SEEDS (0 FAILURES)!")
        print("=" * 80)


if __name__ == "__main__":
    main()


