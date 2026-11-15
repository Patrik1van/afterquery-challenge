"""Extract shot trajectories, monitor readings, and agent text from Harbor jobs.

Outputs to runs/harbor/<agent>/<task>_<attempt>.json for webapp replay and evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re

REPO_ROOT = Path(__file__).resolve().parent.parent


def clean_agent_name(name: str) -> str:
    """Normalize agent name to clean directory name."""
    if not name:
        return "unknown"
    if "StopEarlyAgent" in name or "stop-early" in name.lower():
        return "stop-early"
    if "opencode" in name.lower():
        return "opencode"
    if "oracle" in name.lower():
        return "oracle"
    if "nop" in name.lower():
        return "nop"
    return name.replace(":", "_").replace("/", "_").lower()


def clean_task_name(name: str) -> str:
    """Strip org prefix from task name (e.g. afterquery/beam-steering-s000 -> beam-steering-s000)."""
    if not name:
        return "unknown"
    return name.split("/")[-1]


def parse_opencode_logs(agent_dir: Path) -> tuple[str, list[dict]]:
    """Parse opencode.txt JSON stream into aggregated agent text and event list."""
    log_file = agent_dir / "opencode.txt"
    if not log_file.exists():
        return "", []

    events: list[dict] = []
    text_chunks: list[str] = []

    try:
        with open(log_file, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith("{"):
                    continue
                try:
                    ev = json.loads(line)
                    events.append(ev)
                    ev_type = ev.get("type")
                    part = ev.get("part", {})
                    if ev_type == "text" and isinstance(part, dict) and "text" in part:
                        text_chunks.append(part["text"])
                    elif ev_type == "reasoning" and isinstance(part, dict) and "text" in part:
                        text_chunks.append(f"[Reasoning]: {part['text']}")
                except Exception:
                    continue
    except Exception as e:
        print(f"Error reading opencode log {log_file}: {e}")

    return "\n\n".join(text_chunks), events


def export_from_jobs(jobs_dir: Path, output_base: Path) -> list[Path]:
    """Scan jobs_dir and export all valid trials with unique attempt numbering per task."""
    output_base.mkdir(parents=True, exist_ok=True)
    exported_files: list[Path] = []

    if not jobs_dir.exists():
        print(f"Jobs directory {jobs_dir} does not exist.")
        return exported_files

    job_dirs = sorted([d for d in jobs_dir.iterdir() if d.is_dir()])
    for jd in job_dirs:
        # Group trials in this job by (agent_name, task_name)
        trials_by_agent_task: dict[tuple[str, str], list[dict]] = {}

        for trial_dir in jd.iterdir():
            if not trial_dir.is_dir() or trial_dir.name.startswith("."):
                continue

            config_file = trial_dir / "config.json"
            task_raw = None
            agent_raw = None

            if config_file.exists():
                try:
                    with open(config_file, "r", encoding="utf-8") as f:
                        tc = json.load(f)
                    task_raw = tc.get("task", {}).get("name")
                    agent_raw = tc.get("agent", {}).get("name") or tc.get("agent", {}).get("import_path")
                except Exception:
                    pass

            if not task_raw:
                match = re.match(r"(beam-steering-s\d{3})", trial_dir.name)
                task_raw = match.group(1) if match else "unknown"

            agent_name = clean_agent_name(agent_raw or "")
            task_name = clean_task_name(task_raw)

            # Read started_at from result.json if present
            started_at = ""
            result_file = trial_dir / "result.json"
            if result_file.exists():
                try:
                    with open(result_file, "r", encoding="utf-8") as f:
                        tr = json.load(f)
                    started_at = tr.get("started_at", "")
                except Exception:
                    pass

            key = (agent_name, task_name)
            if key not in trials_by_agent_task:
                trials_by_agent_task[key] = []

            trials_by_agent_task[key].append({
                "trial_dir": trial_dir,
                "agent_raw": agent_raw,
                "agent_name": agent_name,
                "task_name": task_name,
                "started_at": started_at,
            })

        # Process each group and assign chronological attempt numbers (1, 2, 3...)
        for (agent_name, task_name), trial_items in trials_by_agent_task.items():
            trial_items.sort(key=lambda x: x["started_at"] or x["trial_dir"].name)

            for attempt, item in enumerate(trial_items, 1):
                trial_dir = item["trial_dir"]
                agent_raw = item["agent_raw"]

                # Read details.json
                details_file = trial_dir / "verifier" / "details.json"
                details: dict = {}
                if details_file.exists():
                    try:
                        with open(details_file, "r", encoding="utf-8") as f:
                            details = json.load(f)
                    except Exception:
                        pass

                # Read reward from reward.txt if details doesn't have it
                reward = details.get("reward")
                reward_file = trial_dir / "verifier" / "reward.txt"
                if reward is None and reward_file.exists():
                    try:
                        reward = float(reward_file.read_text().strip())
                    except Exception:
                        reward = 0.0

                # Read agent logs / text
                agent_dir = trial_dir / "agent"
                agent_text, events = "", []
                if agent_dir.exists():
                    if (agent_dir / "opencode.txt").exists():
                        agent_text, events = parse_opencode_logs(agent_dir)
                    else:
                        for txt_file in agent_dir.glob("*.txt"):
                            if txt_file.name != "exit-code.txt":
                                try:
                                    agent_text += txt_file.read_text(encoding="utf-8", errors="replace") + "\n"
                                except Exception:
                                    pass

                trajectory_doc = {
                    "agent": agent_name,
                    "agent_raw": agent_raw,
                    "task": task_name,
                    "attempt": attempt,
                    "trial_name": trial_dir.name,
                    "job_id": jd.name,
                    "reward": reward,
                    "solved": details.get("solved", False),
                    "shots_used": details.get("shots_used", 0),
                    "budget": details.get("budget", 60),
                    "s0": details.get("s0"),
                    "sf": details.get("sf"),
                    "progress": details.get("progress"),
                    "is_lost": details.get("is_lost"),
                    "loss_s": details.get("loss_s"),
                    "shots": details.get("history", []),
                    "true_trajectory": details.get("true_trajectory", {}),
                    "agent_text": agent_text.strip(),
                    "events_count": len(events),
                }

                out_dir = output_base / agent_name
                out_dir.mkdir(parents=True, exist_ok=True)
                out_file = out_dir / f"{task_name}_{attempt}.json"

                with open(out_file, "w", encoding="utf-8", newline="\n") as f:
                    json.dump(trajectory_doc, f, indent=2)

                exported_files.append(out_file)

    return exported_files


def main():
    parser = argparse.ArgumentParser(description="Export Harbor trajectories to replay JSON format.")
    parser.add_argument("--jobs-dir", type=Path, default=REPO_ROOT / "jobs", help="Path to Harbor jobs directory")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "runs" / "harbor", help="Output directory")
    args = parser.parse_args()

    files = export_from_jobs(args.jobs_dir, args.output_dir)
    print(f"\n[Export Completed] Exported {len(files)} trial trajectories into {args.output_dir}:")
    for f in files:
        print(f"  - {f.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

