"""Score an alien-arithmetic v1 run.

Runs in a separate verifier container. It sees only:
  /app/answer.json             the agent's answer (artifact from the agent's container)
  /var/log/lab/queries.jsonl   every query the lab answered (artifact from the lab sidecar)
  /tests/answers.json          the puzzles and reference solutions (baked into this image only)

An answer is a full expression (operators and optional parentheses). It is
correct if it uses the puzzle's numbers in order with exactly three operators
and evaluates to the target under the lab's semantics. Any such expression
counts, not only the reference solutions.

reward = (# puzzles solved) / (# puzzles), written to /logs/verifier/reward.txt.
Diagnostics go to /logs/verifier/details.json. They never affect the reward.
"""

import json
import os
import re
import sys
from pathlib import Path

ANSWER = Path(os.environ.get("ANSWER_PATH", "/app/answer.json"))
QUERY_LOG = Path(os.environ.get("QUERY_LOG_PATH", "/var/log/lab/queries.jsonl"))
EXPECTED = Path(os.environ.get("EXPECTED_PATH", "/tests/answers.json"))
OUT_DIR = Path(os.environ.get("VERIFIER_OUT", "/logs/verifier"))

# Must match environment/lab/server.py.
TRUE = {
    "⊕": lambda a, b: a + b + 1,
    "⊗": lambda a, b: a * b - 1,
    "◇": lambda a, b: max(a, b),
    "▽": lambda a, b: abs(a - b),
    "⊘": lambda a, b: 2 * a - b,
    "⊛": lambda a, b: a + b * b,
}
# Diagnostic semantics: what a model computes if it assumed familiar math for
# the old symbols, or treated the non-commutative operators as commutative.
FAMILIAR = {"⊕": lambda a, b: a + b, "⊗": lambda a, b: a * b,
            "◇": lambda a, b: max(a, b), "▽": lambda a, b: a - b}
SWAPPED = dict(TRUE, **{"⊘": lambda a, b: 2 * b - a, "⊛": lambda a, b: b + a * a})

TOKEN_RE = re.compile(r"\s*(\d+|[⊕⊗◇▽⊘⊛()])")


class BadAnswer(ValueError):
    pass


def parse(expr):
    """Return (tree, numbers, n_ops). Tree: int leaves, (op, left, right) nodes."""
    if not isinstance(expr, str):
        raise BadAnswer("answer must be a string")
    tokens, pos, expr = [], 0, expr.strip()
    while pos < len(expr):
        m = TOKEN_RE.match(expr, pos)
        if not m:
            raise BadAnswer("invalid token")
        tokens.append(m.group(1))
        pos = m.end()
    pos = 0

    def peek():
        return tokens[pos] if pos < len(tokens) else None

    def atom():
        nonlocal pos
        tok = peek()
        if tok == "(":
            pos += 1
            node = chain()
            if peek() != ")":
                raise BadAnswer("unbalanced parentheses")
            pos += 1
            return node
        if tok is not None and tok.isdigit():
            pos += 1
            return int(tok)
        raise BadAnswer("expected a number or '('")

    def chain():
        nonlocal pos
        node = atom()
        while peek() in TRUE:
            op = tokens[pos]
            pos += 1
            node = (op, node, atom())
        return node

    tree = chain()
    if pos != len(tokens):
        raise BadAnswer("trailing tokens")
    numbers = [int(t) for t in tokens if t.isdigit()]
    n_ops = sum(1 for t in tokens if t in TRUE)
    return tree, numbers, n_ops


def compute(tree, table):
    """Evaluate under a semantics table; None if it uses an operator the table lacks."""
    if isinstance(tree, int):
        return tree
    op, left, right = tree
    if op not in table:
        return None
    a, b = compute(left, table), compute(right, table)
    if a is None or b is None:
        return None
    return table[op](a, b)


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


def grade(given, spec):
    result = {"given": given, "reference_solutions": spec["reference_solutions"],
              "correct": False, "error": None, "value": None,
              "anchored_on_familiar": False, "assumed_commutative": False}
    if given is None:
        result["error"] = "missing"
        return result
    try:
        tree, numbers, n_ops = parse(given)
    except BadAnswer as e:
        result["error"] = f"unparseable: {e}"
        return result
    if numbers != spec["numbers"] or n_ops != 3:
        result["error"] = "must use the puzzle's numbers in order with exactly three operators"
        return result
    value = compute(tree, TRUE)
    result["value"] = value
    result["correct"] = value == spec["target"]
    if not result["correct"]:
        result["anchored_on_familiar"] = compute(tree, FAMILIAR) == spec["target"]
        result["assumed_commutative"] = compute(tree, SWAPPED) == spec["target"]
    return result


def main():
    expected = json.loads(EXPECTED.read_text(encoding="utf-8"))
    answer, answer_error = load_answer()

    queries = []
    if QUERY_LOG.exists():
        queries = [json.loads(l) for l in QUERY_LOG.read_text(encoding="utf-8").splitlines() if l.strip()]

    per_puzzle = {pid: grade(answer.get(pid), spec) for pid, spec in expected.items()}
    solved = sum(p["correct"] for p in per_puzzle.values())
    reward = solved / len(expected)
    details = {
        "reward": round(reward, 4),
        "solved": solved,
        "total": len(expected),
        "queries_used": len(queries),
        "answer_error": answer_error,
        "anchored_on_familiar": sum(p["anchored_on_familiar"] for p in per_puzzle.values()),
        "assumed_commutative": sum(p["assumed_commutative"] for p in per_puzzle.values()),
        "puzzles": per_puzzle,
        "queries": queries,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "reward.txt").write_text(f"{reward:.4f}\n")
    (OUT_DIR / "details.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_keys = ("reward", "solved", "total", "queries_used",
                    "anchored_on_familiar", "assumed_commutative", "answer_error")
    print(json.dumps({k: details[k] for k in summary_keys}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:  # never leave the run without a reward
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "reward.txt").write_text("0\n")
        print(f"verifier error: {e}", file=sys.stderr)
        sys.exit(1)
