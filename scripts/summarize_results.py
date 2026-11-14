"""Summarize Harbor evaluation runs into a results table and write results/eval_summary.md."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent


def summarize(runs_dir: Path, output_file: Path | None = None) -> str:
    """Read exported runs and produce markdown summary table."""
    agents_order = ["oracle", "opencode", "stop-early", "nop"]
    tasks_order = [f"beam-steering-s{i:03d}" for i in range(5)]

    # data[agent][task] = list of rewards
    data: dict[str, dict[str, list[float]]] = {}
    solved_counts: dict[str, list[bool]] = {}

    for agent_dir in runs_dir.iterdir():
        if not agent_dir.is_dir():
            continue
        agent = agent_dir.name
        if agent not in data:
            data[agent] = {t: [] for t in tasks_order}
            solved_counts[agent] = []

        for f in agent_dir.glob("*.json"):
            try:
                with open(f, "r", encoding="utf-8") as fp:
                    doc = json.load(fp)
                task = doc.get("task", "")
                reward = float(doc.get("reward", 0.0))
                solved = bool(doc.get("solved", False))
                if task in data[agent]:
                    data[agent][task].append(reward)
                    solved_counts[agent].append(solved)
            except Exception:
                continue

    # Order agents: defined order first, then any extra
    all_agents = [a for a in agents_order if a in data] + [a for a in data if a not in agents_order]

    lines = []
    lines.append("# CERN TT20 Beam Steering Benchmark — Harbor Evaluation Summary\n")
    lines.append(f"Evaluated with Harbor framework on Modal cloud across {len(tasks_order)} synthetic lattice seeds.\n")

    # Header
    header = ["Agent"] + [f"Task s{i:03d}" for i in range(5)] + ["Overall (Mean ± Std)", "Solved %"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join([":---"] + [":---:"] * (len(header) - 1)) + " |")

    for agent in all_agents:
        row = [f"**{agent}**"]
        all_rewards = []
        for task in tasks_order:
            rews = data[agent].get(task, [])
            if not rews:
                row.append("—")
            elif len(rews) == 1:
                row.append(f"{rews[0]:.3f}")
                all_rewards.append(rews[0])
            else:
                row.append(f"{np.mean(rews):.3f} ± {np.std(rews):.3f}")
                all_rewards.extend(rews)

        if all_rewards:
            overall_str = f"{np.mean(all_rewards):.3f} ± {np.std(all_rewards):.3f}"
            solv_list = solved_counts.get(agent, [])
            solv_pct = (sum(1 for s in solv_list if s) / len(solv_list) * 100.0) if solv_list else 0.0
            row.append(overall_str)
            row.append(f"{solv_pct:.1f}%")
        else:
            row.extend(["—", "—"])

        lines.append("| " + " | ".join(row) + " |")

    summary_md = "\n".join(lines) + "\n"

    if output_file:
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8", newline="\n") as fp:
            fp.write(summary_md)

    return summary_md


def main():
    parser = argparse.ArgumentParser(description="Summarize Harbor evaluation runs.")
    parser.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "runs" / "harbor", help="Runs directory")
    parser.add_argument("--output", type=Path, default=REPO_ROOT / "results" / "eval_summary.md", help="Output MD file")
    args = parser.parse_args()

    table = summarize(args.runs_dir, args.output)
    print("\n" + table)
    print(f"Summary written to {args.output.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
