"""Grade a Falsify run and write the reward and diagnostics.

Inputs (artifacts copied into the separate verifier container):
    /app/answer.json             the agent's answers
    /var/log/lab/queries.jsonl   every query the lab answered
    /var/log/lab/model.json      the operator model committed before the puzzles were revealed
    /var/log/lab/puzzles.json    the puzzles the lab posed
    /tests/semantics.json        the true operators (only in this image)

Reward: max(0, solved - 0.5 * wrong "none") / number of puzzles, written to
/logs/verifier/reward.txt. Solvability and solutions are computed by brute force under
the true operators; any expression that reaches the target is accepted. Diagnostics in
/logs/verifier/details.json never affect the reward.
"""

import itertools
import json
import os
import re
import sys
from pathlib import Path

from formula import BadFormula, apply, compile_formula

ANSWER = Path(os.environ.get("ANSWER_PATH", "/app/answer.json"))
QUERY_LOG = Path(os.environ.get("QUERY_LOG_PATH", "/var/log/lab/queries.jsonl"))
MODEL_LOG = Path(os.environ.get("MODEL_LOG_PATH", "/var/log/lab/model.json"))
PUZZLES = Path(os.environ.get("PUZZLES_PATH", "/var/log/lab/puzzles.json"))
OUT_DIR = Path(os.environ.get("VERIFIER_OUT", "/logs/verifier"))

WRONG_NONE_PENALTY = 0.5
SEMANTICS = json.loads(Path(__file__).with_name("semantics.json").read_text(encoding="utf-8"))
TOKEN_RE = re.compile(r"\s*(\d+|[" + "".join(SEMANTICS) + "()])")
GRID = [0, 1, 2, 3, 5, 7, 10, 13, 50, 100, 200, 999]
N_VALUES = [2, 3, 4]
SHAPES = ["{0} {a} {1} {b} {2} {c} {3}", "({0} {a} ({1} {b} {2})) {c} {3}", "({0} {a} {1}) {b} ({2} {c} {3})",
          "{0} {a} (({1} {b} {2}) {c} {3})", "{0} {a} ({1} {b} ({2} {c} {3}))"]


class BadAnswer(ValueError):
    """Raised for an answer that is not a valid expression."""


def load_model():
    """Return the committed model and the budget used before reveal, or (None, None)."""
    if not MODEL_LOG.exists():
        return None, None
    data = json.loads(MODEL_LOG.read_text(encoding="utf-8") or "null")
    if not isinstance(data, dict):
        return None, None
    return data.get("model"), data.get("budget_used_before_reveal")


def model_table(model):
    """Compile formulas into callables.

    Args:
        model: Mapping of symbol to formula string.

    Returns:
        Mapping of symbol to f(a, b, n, i, d) that returns None on an arithmetic error;
        symbols with an invalid formula are left out.
    """
    table = {}
    for sym, src in (model or {}).items():
        if sym not in SEMANTICS:
            continue
        try:
            node = compile_formula(src)
        except BadFormula:
            continue

        def f(a, b, n, i, d, node=node):
            try:
                return apply(node, a, b, n, i, d)
            except (ArithmeticError, TypeError, ValueError):
                return None
        table[sym] = f
    return table


TRUE = model_table({s: v["rule"] for s, v in SEMANTICS.items()})
NAIVE = model_table({s: v["naive"] for s, v in SEMANTICS.items()})


def contexts():
    """Yield every (n, i, d) an operation can have in an expression of 2 to 4 operands."""
    for n in N_VALUES:
        for i in range(1, n):
            for d in range(0, n - 1):
                yield n, i, d


def score_model(model):
    """Return, per symbol, whether the committed formula matches the true operator on the test grid."""
    table = model_table(model)
    return {sym: sym in table and all(table[sym](a, b, n, i, d) == TRUE[sym](a, b, n, i, d)
                                      for n, i, d in contexts() for a in GRID for b in GRID)
            for sym in TRUE}


def compute_model(tree, mtable, n):
    """Evaluate an expression tree under a table of operators, or return None on an error."""
    position = 0

    def go(t, depth):
        nonlocal position
        if isinstance(t, int):
            return t
        op, left, right = t
        a = go(left, depth + 1)
        position += 1
        i = position
        b = go(right, depth + 1)
        if a is None or b is None or op not in mtable:
            return None
        return mtable[op](a, b, n, i, depth)
    return go(tree, 0)


def solutions_under(table, numbers, target):
    """Yield every operator choice and bracketing of the numbers that reaches the target."""
    for o in itertools.product(list(table), repeat=3):
        for shape in SHAPES:
            expr = shape.format(*numbers, a=o[0], b=o[1], c=o[2])
            if compute_model(parse(expr)[0], table, len(numbers)) == target:
                yield expr


def solvable_under(table, numbers, target):
    """Return whether any expression reaches the target under the given operators."""
    return next(solutions_under(table, numbers, target), None) is not None


def parse(expr):
    """Parse an answer expression.

    Args:
        expr: The answer string.

    Returns:
        A (tree, numbers in order, operator count) tuple.

    Raises:
        BadAnswer: If the expression is malformed.
    """
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


def load_answer():
    """Return (answers dict, error message); a malformed answer file yields an empty dict."""
    if not ANSWER.exists():
        return {}, "answer file missing"
    try:
        data = json.loads(ANSWER.read_text(encoding="utf-8"))
    except Exception as e:
        return {}, f"answer is not valid JSON: {e}"
    if not isinstance(data, dict):
        return {}, "answer must be a JSON object"
    return data, None


def is_none(given):
    """Return whether an answer declares the puzzle unsolvable."""
    return isinstance(given, str) and given.strip().lower() == "none"


def grade(given, spec, agent_table=None):
    """Grade one answer.

    Args:
        given: The agent's answer: an expression, "none" or None if missing.
        spec: The puzzle with "numbers", "target", "solvable" and "reference_solutions".
        agent_table: The agent's committed model as callables, used for diagnostics only.

    Returns:
        A dict with "correct" and diagnostic fields.
    """
    solvable = spec["solvable"]
    result = {"given": given, "solvable": solvable, "reference_solutions": spec["reference_solutions"],
              "correct": False, "error": None, "value": None,
              "false_none": False, "false_solution": False, "naive_reading": False}
    if given is None:
        result["error"] = "missing"
        return result
    if is_none(given):
        result["correct"] = not solvable
        result["false_none"] = solvable
        if agent_table is not None:
            result["consistent_with_model"] = not solvable_under(agent_table, spec["numbers"], spec["target"])
        return result
    if not solvable:
        result["false_solution"] = True
    try:
        tree, numbers, n_ops = parse(given)
    except BadAnswer as e:
        result["error"] = f"unparseable: {e}"
        return result
    if numbers != spec["numbers"] or n_ops != 3:
        result["error"] = "must use the puzzle's numbers in order with exactly three operators"
        return result
    value = compute_model(tree, TRUE, len(numbers))
    result["value"] = value
    if agent_table is not None:
        result["consistent_with_model"] = compute_model(tree, agent_table, len(spec["numbers"])) == spec["target"]
    result["correct"] = value == spec["target"]
    if not result["correct"]:
        result["naive_reading"] = compute_model(tree, NAIVE, len(numbers)) == spec["target"]
    return result


def main():
    """Grade the run and write reward.txt and details.json."""
    expected = {}
    for pid, p in json.loads(PUZZLES.read_text(encoding="utf-8")).items():
        refs = list(solutions_under(TRUE, p["numbers"], p["target"]))
        expected[pid] = {**p, "solvable": bool(refs), "reference_solutions": refs}
    answer, answer_error = load_answer()

    queries = []
    if QUERY_LOG.exists():
        queries = [json.loads(line) for line in QUERY_LOG.read_text(encoding="utf-8").splitlines() if line.strip()]

    model, budget_before_reveal = load_model()
    model_ok = score_model(model) if model else {}
    full_table = model_table(model) if model else {}
    agent_table = full_table if len(full_table) == len(TRUE) else None
    per_puzzle = {pid: grade(answer.get(pid), spec, agent_table) for pid, spec in expected.items()}
    solved = sum(p["correct"] for p in per_puzzle.values())
    wrong_none = sum(p["false_none"] for p in per_puzzle.values())
    reward = max(0.0, solved - WRONG_NONE_PENALTY * wrong_none) / len(expected)
    details = {
        "reward": round(reward, 4),
        "solved": solved,
        "total": len(expected),
        "queries_used": len(queries),
        "budget_spent": sum(q.get("cost", 1) for q in queries),
        "answer_error": answer_error,
        "naive_reading": sum(p["naive_reading"] for p in per_puzzle.values()),
        "model_committed": model is not None,
        "budget_used_before_reveal": budget_before_reveal,
        "model_operators_correct": sum(model_ok.values()),
        "model_wrong_operators": [s for s, ok in model_ok.items() if not ok],
        "model": model,
        "answers_inconsistent_with_model": sum(p.get("consistent_with_model") is False for p in per_puzzle.values()),
        "false_none": wrong_none,
        "false_solution": sum(p["false_solution"] for p in per_puzzle.values()),
        "puzzles": per_puzzle,
        "queries": queries,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "reward.txt").write_text(f"{reward:.4f}\n")
    (OUT_DIR / "details.json").write_text(json.dumps(details, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_keys = ("reward", "solved", "total", "queries_used", "budget_spent",
                    "model_committed", "model_operators_correct", "answers_inconsistent_with_model",
                    "false_none", "false_solution", "naive_reading", "answer_error")
    print(json.dumps({k: details[k] for k in summary_keys}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        (OUT_DIR / "reward.txt").write_text("0\n")
        print(f"verifier error: {e}", file=sys.stderr)
        sys.exit(1)
