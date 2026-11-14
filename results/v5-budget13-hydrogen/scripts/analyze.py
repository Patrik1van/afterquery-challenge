"""Per-trial and aggregate statistics for the Hydrogen runs listed in runs.tsv."""
import json
import statistics as st
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path("/Users/patrikivan/afterquery-challenge")
SCR = Path(__file__).resolve().parent.parent  # the results folder: runs.tsv and trials.json live here
sys.path.insert(0, str(ROOT / "tasks/alien-arithmetic-v5/environment/lab"))
import server as S  # noqa: E402  (parser + semantics only)

ASYM = ["◇", "▽", "△", "⊘", "⊛"]


def walk(t, depth=0, n=None, out=None):
    """Collect (op, a_value, b_value, n) for every operation in a parsed query (values under the true rules)."""
    out = [] if out is None else out
    if isinstance(t, int):
        return t
    a = walk(t[1], depth + 1, n, out)
    b = walk(t[2], depth + 1, n, out)
    out.append((t[0], a, b, n))
    try:
        return S.apply(S.OPS[t[0]], a, b, n, 1, depth)
    except Exception:
        return None


def probe_features(queries):
    seen = {}
    for q in queries:
        try:
            tree = S.parse(q["expr"])
        except Exception:
            continue
        ops = []
        walk(tree, n=S.operands(tree), out=ops)
        for op, a, b, n in ops:
            if a is None or b is None:
                continue
            f = seen.setdefault(op, {"a<b": False, "a>b": False, "sum>=256": False, "negative_in": False, "n>=3": False})
            f["a<b"] |= a < b
            f["a>b"] |= a > b
            f["sum>=256"] |= a + b >= 256
            f["negative_in"] |= a < 0 or b < 0
            f["n>=3"] |= n >= 3
    return seen


def ts(x):
    return datetime.fromisoformat(x.replace("Z", "+00:00"))


def trial_stats(d):
    det_p = d / "verifier/details.json"
    det = json.loads(det_p.read_text()) if det_p.exists() else {}
    res = json.loads((d / "result.json").read_text())
    exc = (d / "exception.txt").read_text() if (d / "exception.txt").exists() else ""
    traj_p = d / "agent/trajectory.json"
    steps = json.loads(traj_p.read_text())["steps"] if traj_p.exists() else []
    calls = [c for s in steps for c in (s.get("tool_calls") or [])]
    cmds = [json.dumps(c.get("arguments", {}), ensure_ascii=False) for c in calls]
    reveal_t = next((s["timestamp"] for s in steps for c in (s.get("tool_calls") or [])
                     if "reveal" in json.dumps(c.get("arguments", {})) and "evaluate" not in json.dumps(c.get("arguments", {}))), None)
    stamps = [s["timestamp"] for s in steps if s.get("timestamp")]
    start = stamps[0] if stamps else None
    end = stamps[-1] if stamps else None
    queries = det.get("queries", [])
    feats = probe_features(queries)
    cost = sum((s.get("metrics") or {}).get("cost_usd") or 0 for s in steps)
    return {
        "trial": d.name,
        "reward": det.get("reward", 0.0),
        "solved": det.get("solved", 0),
        "timeout": "AgentTimeoutError" in exc,
        "error": exc.strip().splitlines()[-1][:120] if exc else None,
        "model_committed": det.get("model_committed"),
        "model_ok": det.get("model_operators_correct"),
        "wrong_ops": {s: (det.get("model") or {}).get(s) for s in det.get("model_wrong_operators", [])},
        "false_none": det.get("false_none"),
        "false_solution": det.get("false_solution"),
        "inconsistent": det.get("answers_inconsistent_with_model"),
        "budget_spent": det.get("budget_spent"),
        "n_queries": len(queries),
        "multi_op_queries": sum(q.get("cost", 1) > 1 for q in queries),
        "overflow_queries": sum(q.get("result") is None for q in queries),
        "first_probe": queries[0]["expr"] if queries else None,
        "queries": [f'{q["expr"]} = {q["result"]}' for q in queries],
        "probe": feats,
        "minutes_total": round((ts(end) - ts(start)).total_seconds() / 60, 1) if start else None,
        "minutes_to_reveal": round((ts(reveal_t) - ts(start)).total_seconds() / 60, 1) if reveal_t else None,
        "steps": len(steps),
        "used_python": any("python" in c for c in cmds),
        "cost_usd": round(cost, 3),
        "puzzles": {p: {"given": v["given"], "correct": v["correct"], "solvable": v["solvable"],
                        "naive": v.get("naive_reading"), "consistent": v.get("consistent_with_model")}
                    for p, v in det.get("puzzles", {}).items()},
    }


def main():
    rows = []
    runs = Path(sys.argv[1]) if len(sys.argv) > 1 else SCR / "runs.tsv"
    for line in runs.read_text().splitlines():
        label, job = line.split("\t")
        job = ROOT / job
        if not label.startswith("hydrogen"):
            o = list(Path(job).glob("alien*/verifier/test-stdout.txt"))
            print(label, o[0].read_text()[:120] if o else "no verifier output")
            continue
        for d in sorted(Path(job).glob("alien*/")):
            r = trial_stats(d)
            r["set"] = label.split("_", 1)[1]
            rows.append(r)
    by = {}
    for r in rows:
        by.setdefault(r["set"], []).append(r)
    print(f"\n{'set':6} {'n':>2} {'mean':>6} {'sd':>5} {'min':>5} {'max':>5} {'=1.0':>4} {'TO':>3}  rewards")
    allr = [r["reward"] for r in rows]
    for k, rs in list(by.items()) + [("ALL", rows)]:
        rw = [r["reward"] for r in rs]
        print(f"{k:6} {len(rw):2} {st.mean(rw):6.3f} {st.pstdev(rw):5.3f} {min(rw):5.2f} {max(rw):5.2f} "
              f"{sum(x == 1 for x in rw):4} {sum(r['timeout'] for r in rs):3}  {rw}")
    print("\nper trial:")
    for r in rows:
        p = r["probe"]
        print(f"{r['set']:5} {r['trial'][-7:]} R={r['reward']:.3f} model={r['model_ok']} wrong={r['wrong_ops']} "
              f"fn={r['false_none']} fs={r['false_solution']} inc={r['inconsistent']} q={r['n_queries']} multi={r['multi_op_queries']} "
              f"ovf={r['overflow_queries']} t_rev={r['minutes_to_reveal']} t={r['minutes_total']} py={r['used_python']} "
              f"$={r['cost_usd']} TO={r['timeout']}")
        print("      △a<b:", p.get("△", {}).get("a<b"), " ◇a<b:", p.get("◇", {}).get("a<b"), " ▽a<b:", p.get("▽", {}).get("a<b"),
              " ⊞sum>=256:", p.get("⊞", {}).get("sum>=256"), " ⊞neg:", p.get("⊞", {}).get("negative_in"),
              " ⊠n>=3:", p.get("⊠", {}).get("n>=3"), " first:", r["first_probe"])
        print("      queries:", " | ".join(r["queries"]))
    wrong = {}
    for r in rows:
        for s, f in r["wrong_ops"].items():
            wrong.setdefault(s, []).append(f)
    print("\nwrong operators across trials:", {s: (len(v), sorted(set(map(str, v)))) for s, v in wrong.items()})
    print("mean reward", round(st.mean(allr), 3))


if __name__ == "__main__":
    main()
