"""Collect the v5 budget-13 Gemini 4 Hydrogen runs (fixed puzzles + seeds 1-4) into this folder.

Usage (from the repo root):  python3 results/v5-budget13-hydrogen/scripts/aggregate.py

Reads runs.tsv (label -> jobs/<timestamp>/), copies each trial's evidence into trials/ and oracle/,
and writes flat tables for analysis:
  trials.json   one record per Hydrogen trial (outcome, committed model, probe coverage, timing, cost)
  trials.csv    the same, flattened
  queries.csv   every lab query: set, trial, order, expression, result, cost
  puzzles.csv   every graded puzzle answer: set, trial, puzzle, numbers, target, solvable, given, correct, ...
  operators.csv one row per (trial, operator): committed formula, correct?, which discriminating probes were run
"""

import csv
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent
ROOT = OUT.parents[1]
sys.path.insert(0, str(HERE))
from analyze import trial_stats  # noqa: E402

# Per trial, keep the evidence and the cleaned transcript; skip opencode's internal db/log (duplicates).
KEEP = ["result.json", "trial.log", "exception.txt", "verifier/details.json", "verifier/test-stdout.txt",
        "verifier/reward.txt", "artifacts", "agent/trajectory.json"]
OPS = ["⊕", "⊗", "◇", "▽", "⊘", "⊛", "⊞", "⊠", "△"]
TRUE = json.loads((ROOT / "tasks/alien-arithmetic-v5/tests/semantics.json").read_text(encoding="utf-8"))


def copy_trial(src, dst):
    for rel in KEEP:
        s = src / rel
        if s.is_dir():
            shutil.copytree(s, dst / rel, dirs_exist_ok=True)
        elif s.exists():
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, dst / rel)


def main():
    rows, queries, puzzles, operators = [], [], [], []
    for line in (OUT / "runs.tsv").read_text().splitlines():
        label, job = line.split("\t")
        kind, pset = label.split("_", 1)
        job = ROOT / job
        for src in sorted(job.glob("alien*/")):
            dst = OUT / ("trials" if kind == "hydrogen" else "oracle") / pset / src.name.split("__")[-1]
            copy_trial(src, dst)
            if kind != "hydrogen":
                continue
            r = trial_stats(src)
            r["set"] = pset
            r["job"] = str(job.relative_to(ROOT))
            rows.append(r)
            tid = r["trial"].split("__")[-1]
            det = json.loads((src / "verifier/details.json").read_text())
            for k, q in enumerate(det.get("queries", []), 1):
                queries.append({"set": pset, "trial": tid, "order": k, "expr": q["expr"], "result": q["result"],
                                "cost": q.get("cost"), "budget_used": q.get("budget_used")})
            posed = json.loads((src / "artifacts/var/log/lab/puzzles.json").read_text(encoding="utf-8"))
            for pid, p in det.get("puzzles", {}).items():
                puzzles.append({"set": pset, "trial": tid, "puzzle": pid,
                                "numbers": " ".join(map(str, posed[pid]["numbers"])), "target": posed[pid]["target"],
                                "solvable": p["solvable"], "given": p["given"], "correct": p["correct"],
                                "false_none": p["false_none"], "false_solution": p["false_solution"],
                                "naive_reading": p.get("naive_reading"), "consistent_with_model": p.get("consistent_with_model"),
                                "error": p.get("error"), "reference_solutions": " | ".join(p.get("reference_solutions", []))})
            model = det.get("model") or {}
            for s in OPS:
                probe = r["probe"].get(s, {})
                operators.append({"set": pset, "trial": tid, "operator": s, "true_rule": TRUE[s]["rule"],
                                  "committed": model.get(s), "correct": bool(model) and s not in det.get("model_wrong_operators", []),
                                  "probed": bool(probe), **{f"tested_{k}": v for k, v in probe.items()}})

    (OUT / "trials.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    flat_keys = ["set", "trial", "job", "reward", "solved", "timeout", "model_committed", "model_ok", "wrong_ops",
                 "false_none", "false_solution", "inconsistent", "budget_spent", "n_queries", "multi_op_queries",
                 "overflow_queries", "first_probe", "minutes_to_reveal", "minutes_total", "steps", "used_python", "cost_usd"]
    write_csv(OUT / "trials.csv", [{k: (json.dumps(r[k], ensure_ascii=False) if isinstance(r[k], dict) else r[k])
                                    for k in flat_keys} for r in rows])
    write_csv(OUT / "queries.csv", queries)
    write_csv(OUT / "puzzles.csv", puzzles)
    write_csv(OUT / "operators.csv", operators)
    print(f"{len(rows)} trials, {len(queries)} queries, {len(puzzles)} puzzle answers, {len(operators)} operator rows")


def write_csv(path, rows):
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


main()
