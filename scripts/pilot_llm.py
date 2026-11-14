"""Gemini function-calling agent for the TT20 beamline environment.

Uses google-genai SDK, python-dotenv, and instruction_draft.md system prompt.
Available tools:
- get_machine_info
- set_correctors
- get_history
- submit
- run_python (isolated sandbox without beamsim)
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any

from dotenv import load_dotenv

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from google import genai
from google.genai import types

from agents.bayes import BayesAgent
from beamsim.env_api import BeamlineEnv, evaluate_env, get_env_true_trajectory


def make_isolated_run_python():
    """Create isolated run_python tool without access to project modules."""
    def run_python(code: str) -> str:
        """Execute Python code in an isolated temporary directory without access to beamsim.

        Args:
            code: Complete standalone Python code string to execute.
        Returns:
            The standard output and error from running the code.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            script_path = Path(temp_dir) / "sandbox_script.py"
            script_path.write_text(code, encoding="utf-8")

            clean_env = os.environ.copy()
            clean_env.pop("PYTHONPATH", None)

            try:
                proc = subprocess.run(
                    [sys.executable, str(script_path)],
                    cwd=temp_dir,
                    env=clean_env,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                output = proc.stdout
                if proc.stderr:
                    output += "\n[stderr]:\n" + proc.stderr
                if proc.returncode != 0:
                    output += f"\n[Process exited with return code {proc.returncode}]"
                return output if output.strip() else "[Execution finished with no output]"
            except subprocess.TimeoutExpired:
                return "[Error: Code execution timed out after 30 seconds]"
            except Exception as e:
                return f"[Error executing code: {e}]"

    return run_python


def run_gemini_episode(
    seed: int,
    budget: int = 60,
    model_name: str = "gemini-3.8-flash",
    api_key: str | None = None,
) -> dict[str, Any]:
    """Run an episode with Gemini function-calling agent."""
    t0 = time.perf_counter()
    agent_subfolder = "gemini_pro" if "pro" in model_name.lower() else "gemini"
    out_dir = repo_root / "runs" / agent_subfolder
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"seed_{seed:03d}.json"

    if out_file.exists():
        try:
            with open(out_file, "r", encoding="utf-8") as f:
                cached = json.load(f)
            sc = cached["evaluation"]
            category = "SOLVED" if sc["solved"] else ("CLOSE_UNDER_095" if sc["sf"] >= 0.80 else ("NEVER_FOUND" if sc["sf"] < 0.50 else "BUDGET_EXHAUSTED"))
            print(f"\n[Gemini Seed {seed}] Loaded cached run: Sf={sc['sf']:.4f}, shots={sc['shots_used']}", flush=True)
            return {
                "seed": seed,
                "reward": sc["reward"],
                "sf": sc["sf"],
                "shots": sc["shots_used"],
                "solved": sc["solved"],
                "elapsed": cached["metadata"].get("elapsed_seconds", 0.0),
                "category": category,
            }
        except Exception:
            pass

    env = BeamlineEnv(seed=seed, budget=budget, randomization="medium")

    # Tool definitions wrapping this episode's env
    def get_machine_info() -> dict[str, Any]:
        """Return the nominal lattice, response matrices, apertures, noise level nu, and budget."""
        return env.get_machine_info()

    def set_correctors(h: list[float], v: list[float]) -> dict[str, Any]:
        """Command absolute corrector angles (10 horizontal, 12 vertical). Costs 1 shot.

        Returns noisy SEM monitor readings {ratio, level} per monitor and shots remaining.
        """
        return env.set_correctors(h, v)

    def get_history() -> list[dict[str, Any]]:
        """Return chronological history of all settings and monitor readings so far."""
        return env.get_history()

    def submit() -> dict[str, Any]:
        """Freeze the episode when the beam is centered. No further commands allowed."""
        return env.submit()

    run_python_tool = make_isolated_run_python()

    # Load operator briefing system prompt
    instr_path = repo_root / "tasks" / "instruction_draft.md"
    system_prompt = instr_path.read_text(encoding="utf-8") if instr_path.exists() else "Control the beamline."

    client = genai.Client(api_key=api_key)

    tools_list = [
        get_machine_info,
        set_correctors,
        get_history,
        submit,
        run_python_tool,
    ]

    tool_dispatch = {
        "get_machine_info": get_machine_info,
        "set_correctors": set_correctors,
        "get_history": get_history,
        "submit": submit,
        "run_python": run_python_tool,
    }

    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        tools=tools_list,
        temperature=0.2,
    )

    chat = client.chats.create(model=model_name, config=config)

    initial_prompt = (
        f"You are starting a new beamline commissioning session on seed {seed} with a budget of {budget} shots. "
        "The line starts unsteered (theta = 0). Start by getting the machine info, investigate the model, "
        "and begin threading and centring the beam using your tools."
    )

    print(f"\n[Gemini Seed {seed}] Starting episode with model: {model_name}...", flush=True)

    prompt = initial_prompt
    max_turns = 12
    turn = 0

    while turn < max_turns and not env._submitted and env._sim.shots_used < budget:
        turn += 1
        try:
            resp = chat.send_message(prompt)
            print(f"  [Turn {turn:02d}] Shots used: {env._sim.shots_used}/{budget} | Submitted: {env._submitted}", flush=True)
            if resp.text:
                first_line = resp.text.strip().split("\n")[0][:100]
                print(f"    Agent: {first_line}", flush=True)
        except Exception as e:
            print(f"  [Turn {turn:02d}] API notice: {e}", flush=True)
            prompt = f"Previous call encountered an issue: {e}. Please continue steering."
            continue

        if env._submitted or env._sim.shots_used >= budget:
            break

        prompt = (
            f"Status: shots used = {env._sim.shots_used}/{budget}. Submitted = {env._submitted}. "
            "If the beam is centered, call submit(). Otherwise, command your next set_correctors or run_python."
        )



    # Ensure episode is submitted if not already
    if not env._submitted:
        env.submit()

    elapsed = time.perf_counter() - t0


    score = evaluate_env(env)
    true_traj = get_env_true_trajectory(env)

    # Save episode log to runs/<agent_subfolder>/seed_XXX.json
    out_dir = repo_root / "runs" / agent_subfolder
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"seed_{seed:03d}.json"

    # Convert env._history to replay shot format
    shots_history = []
    for h_item in env.get_history():
        shots_history.append({
            "shot": h_item["shot"],
            "theta_h": h_item["theta_h"],
            "theta_v": h_item["theta_v"],
            "obs_h": h_item["obs_h"],
            "obs_v": h_item["obs_v"],
            "posterior_mean_h": None,
            "posterior_mean_v": None,
            "p_lost": None,
            "predicted_s_mean": None,
            "action_type": "gemini_step",
        })

    log_data = {
        "metadata": {
            "agent": "gemini",
            "model": model_name,
            "seed": seed,
            "budget": budget,
            "randomization": "medium",
            "elapsed_seconds": round(elapsed, 4),
        },
        "shots": shots_history,
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
        "reward": score.reward,
        "sf": score.sf,
        "shots": score.shots_used,
        "solved": score.solved,
        "elapsed": elapsed,
        "category": category,
    }


def run_bayes_comparison(
    seeds: list[int],
    budget: int = 60,
    n_particles: int = 25_000,
    roughening: float = 0.02,
) -> list[dict[str, Any]]:
    """Run Bayes agent on comparison seeds with roughening 0.02 and budget 60."""
    bayes_results = []
    print(f"\nRunning Bayes comparison agent on seeds {seeds} (roughening={roughening}, budget={budget})...")

    for seed in seeds:
        t0 = time.perf_counter()
        env = BeamlineEnv(seed=seed, budget=budget, randomization="medium")
        agent = BayesAgent(
            n_particles=n_particles,
            chunk_size=n_particles,
            roughening=roughening,
            seed=seed,
        )
        agent_logs = agent.solve(env)
        elapsed = time.perf_counter() - t0

        score = evaluate_env(env)
        true_traj = get_env_true_trajectory(env)

        category = "SOLVED" if score.solved else ("CLOSE_UNDER_095" if score.sf >= 0.80 else ("NEVER_FOUND" if score.sf < 0.50 else "BUDGET_EXHAUSTED"))
        bayes_results.append({
            "seed": seed,
            "reward": score.reward,
            "sf": score.sf,
            "shots": score.shots_used,
            "solved": score.solved,
            "elapsed": elapsed,
            "category": category,
        })
        status = "SOLVED" if score.solved else f"FAILED ({category})"
        print(f"  [Bayes Seed {seed:03d}]: {status:<24} | Sf = {score.sf:.4f} | Shots = {score.shots_used:2d} | Rew = {score.reward:.3f} | {elapsed:.2f}s")

    return bayes_results


def main():
    parser = argparse.ArgumentParser(description="Pilot Gemini function-calling agent for TT20 beamline.")
    parser.add_argument("--model", type=str, default="gemini-3-pro-image", help="Gemini Pro model name.")
    parser.add_argument("--budget", type=int, default=60, help="Shot budget.")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="Seeds to evaluate.")
    parser.add_argument("--parallel", action="store_true", default=True, help="Run seeds in parallel.")
    args = parser.parse_args()

    # Read GEMINI_API_KEY from .env
    env_file = repo_root / ".env"
    load_dotenv(env_file)
    api_key = os.getenv("GEMINI_API_KEY", "").strip()

    if not api_key:
        print("\n" + "=" * 80)
        print("ERROR: GEMINI_API_KEY is not set or empty in .env.")
        print(f"Please paste your Gemini API key into: {env_file}")
        print("=" * 80)
        sys.exit(1)

    print("=" * 80)
    print(f"PILOT RUN: GEMINI PRO IN PARALLEL (Model={args.model}, Seeds={args.seeds}, Budget={args.budget})")
    print("=" * 80)

    # 1. Run Gemini Pro on seeds 0, 1, 2 in parallel using ThreadPoolExecutor (API-bound)
    import concurrent.futures

    def _run_seed_pro(s: int) -> dict[str, Any]:
        return run_gemini_episode(
            seed=s,
            budget=args.budget,
            model_name=args.model,
            api_key=api_key,
        )

    t_pro_start = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(args.seeds)) as executor:
        gemini_pro_results = list(executor.map(_run_seed_pro, args.seeds))
    t_pro_total = time.perf_counter() - t_pro_start
    print(f"Gemini Pro finished in {t_pro_total:.1f}s total wall time.\n")

    # 2. Load Gemini Flash results (from runs/gemini/seed_XXX.json)
    gemini_flash_results = []
    for s in args.seeds:
        flash_file = repo_root / "runs" / "gemini" / f"seed_{s:03d}.json"
        if flash_file.exists():
            with open(flash_file, "r", encoding="utf-8") as f:
                d = json.load(f)
            sc = d["evaluation"]
            gemini_flash_results.append({
                "seed": s,
                "reward": sc["reward"],
                "sf": sc["sf"],
                "shots": sc["shots_used"],
                "solved": sc["solved"],
                "elapsed": d["metadata"].get("elapsed_seconds", 0.0),
            })
        else:
            res = run_gemini_episode(seed=s, budget=args.budget, model_name="gemini-3.8-flash", api_key=api_key)
            gemini_flash_results.append(res)

    # 3. Load or compute Bayes reference on the exact same seeds
    bayes_results = []
    for s in args.seeds:
        bayes_file = repo_root / "runs" / "bayes" / f"seed_{s:03d}.json"
        if bayes_file.exists():
            with open(bayes_file, "r", encoding="utf-8") as f:
                d = json.load(f)
            sc = d["evaluation"]
            bayes_results.append({
                "seed": s,
                "reward": sc["reward"],
                "sf": sc["sf"],
                "shots": sc["shots_used"],
                "solved": sc["solved"],
                "elapsed": d["metadata"].get("elapsed_seconds", 0.0),
            })
        else:
            b_list = run_bayes_comparison(seeds=[s], budget=args.budget, n_particles=25_000, roughening=0.02)
            bayes_results.append(b_list[0])

    # 4. Print side-by-side comparison table per seed
    print("\n" + "=" * 92)
    print("FINAL COMPARISON TABLE: BAYES vs GEMINI FLASH vs GEMINI PRO (PER SEED)")
    print("=" * 92)
    print(f"{'Seed':<6} | {'Agent':<26} | {'Reward':<10} | {'Sf':<10} | {'Shots':<8} | {'Solved':<8} | {'Time':<8}")
    print("-" * 92)

    for s_idx, s in enumerate(args.seeds):
        b = bayes_results[s_idx]
        gf = gemini_flash_results[s_idx]
        gp = gemini_pro_results[s_idx]

        b_solv = "YES" if b["solved"] else "NO"
        gf_solv = "YES" if gf["solved"] else "NO"
        gp_solv = "YES" if gp["solved"] else "NO"

        print(f"{s:<6d} | {'Bayes (roughening 0.02)':<26} | {b['reward']:<10.3f} | {b['sf']:<10.4f} | {b['shots']:<8d} | {b_solv:<8} | {b['elapsed']:<7.1f}s")
        print(f"{s:<6d} | {'Gemini Flash (3.8-flash)':<26} | {gf['reward']:<10.3f} | {gf['sf']:<10.4f} | {gf['shots']:<8d} | {gf_solv:<8} | {gf['elapsed']:<7.1f}s")
        print(f"{s:<6d} | {'Gemini Pro (3-pro-image)':<26} | {gp['reward']:<10.3f} | {gp['sf']:<10.4f} | {gp['shots']:<8d} | {gp_solv:<8} | {gp['elapsed']:<7.1f}s")
        print("-" * 92)

    # 5. Print Summary Row with Mean +/- Std
    print("\n" + "=" * 92)
    print("OVERALL SUMMARY ACROSS SEEDS 0, 1, 2 (MEAN +/- STD)")
    print("=" * 92)
    print(f"{'Agent':<28} | {'Reward (mean +/- std)':<23} | {'Sf (mean +/- std)':<20} | {'Solved %':<10}")
    print("-" * 92)

    agents_data = [
        ("Bayes (roughening 0.02)", bayes_results),
        ("Gemini Flash (3.8-flash)", gemini_flash_results),
        ("Gemini Pro (3-pro-image)", gemini_pro_results),
    ]

    for name, res_list in agents_data:
        r_vals = [x["reward"] for x in res_list]
        sf_vals = [x["sf"] for x in res_list]
        solv_cnt = sum(1 for x in res_list if x["solved"])
        solv_pct = (solv_cnt / len(res_list)) * 100.0

        r_str = f"{np.mean(r_vals):.3f} +/- {np.std(r_vals):.3f}"
        sf_str = f"{np.mean(sf_vals):.4f} +/- {np.std(sf_vals):.4f}"
        print(f"{name:<28} | {r_str:<23} | {sf_str:<20} | {solv_pct:5.1f}%")

    print("=" * 92)
    print("\nEvaluation of seeds 0, 1, 2 completed.")


if __name__ == "__main__":
    main()


