"""Score an alien-arithmetic run.

Runs in a separate verifier container. It sees only:
  /app/answer.json             the agent's answer (artifact from the agent's container)
  /var/log/lab/queries.jsonl   every query the lab answered (artifact from the lab sidecar)
  /tests/answers.json          the expected solutions (baked into this image only)

reward = (# puzzles solved) / (# puzzles), written to /logs/verifier/reward.txt.
Diagnostics go to /logs/verifier/details.json. They never affect the reward.
"""

import json
import os
import sys
from pathlib import Path

ANSWER = Path(os.environ.get("ANSWER_PATH", "/app/answer.json"))
QUERY_LOG = Path(os.environ.get("QUERY_LOG_PATH", "/var/log/lab/queries.jsonl"))
EXPECTED = Path(os.environ.get("EXPECTED_PATH", "/tests/answers.json"))
OUT_DIR = Path(os.environ.get("VERIFIER_OUT", "/logs/verifier"))

SYMBOLS = set("⊕⊗◇▽")


def load_answer():
    """Return (dict, error). Any malformed answer becomes an empty dict."""
    if not ANSWER.exists():
        return {}, "answer file missing"
    try:
        data = json.loads(ANSWER.read_text(encoding="utf-8"))
    except Exception as e:
        return {}, f"answer is not valid JSON: {e}"
    if not isinstance(data, dict):
        return {}, "answer must be a JSON object"
    return data, None


def normalize(value):
    """A valid answer is exactly three operator symbols (whitespace ignored)."""
    if not isinstance(value, str):
        return None
    s = "".join(value.split())
    if len(s) != 3 or any(c not in SYMBOLS for c in s):
        return None
    return s


def main():
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))
    answer, answer_error = load_answer()

    queries = []
    if QUERY_LOG.exists():
        queries = [json.loads(l) for l in QUERY_LOG.read_text(encoding="utf-8").splitlines() if l.strip()]

    per_puzzle, solved = {}, 0
    for pid, spec in expected.items():
        given = normalize(answer.get(pid))
        correct = given == spec["solution"]
        solved += correct
        per_puzzle[pid] = {
            "given": answer.get(pid),
            "expected": spec["solution"],
            "correct": correct,
            # wrong answer that would be right under +, ×, max, − : the model
            # assumed familiar semantics instead of testing them
            "anchored_on_familiar": (not correct) and given in spec["familiar_solutions"],
        }

    reward = solved / len(expected)
    details = {
        "reward": round(reward, 4),
        "solved": solved,
        "total": len(expected),
        "queries_used": len(queries),
        "answer_error": answer_error,
        "anchored_on_familiar": sum(p["anchored_on_familiar"] for p in per_puzzle.values()),
        "puzzles": per_puzzle,
        "queries": queries,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "reward.txt").write_text(f"{reward:.4f}\n")
    (OUT_DIR / "details.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: details[k] for k in ("reward", "solved", "total", "queries_used", "anchored_on_familiar", "answer_error")}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # never leave the run without a reward
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "reward.txt").write_text("0\n")
        print(f"verifier error: {e}", file=sys.stderr)
        sys.exit(1)
