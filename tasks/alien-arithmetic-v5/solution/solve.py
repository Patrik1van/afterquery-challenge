"""Reference solver (the 'oracle') for v5: nine operators, budget 13.

Phase 1, 13 points, every one needed:
  * 9 points: probe each operator once on (900, 13). The sum 913 exceeds 256,
    so a modulo shows up, and almost every candidate gives a distinct value.
    Still ambiguous afterwards: ◇ (max or a), ▽ and △ (a-b, |a-b| or a-b
    truncated at 0), and ⊠ (a+b+2 or a+b+n; with 2 operands n is 2).
  * 3 points: one 4-operand query, (13 ◇ 900) ⊠ (20 ▽ 500). It tests ◇ and ▽
    with a < b and ⊠ at n = 4 at the same time; every combination of the
    remaining hypotheses predicts a different value, so one answer decides all three.
  * 1 point: 13 △ 900 tests △ with a < b.

Commit: write the formulas to /app/model.json and run `reveal`.
Phase 2: brute-force each revealed puzzle (n = 4) over all operator choices
and all five bracketings; answer "none" when nothing fits.
"""

import itertools
import json
import os
import re
import subprocess
from pathlib import Path

SYMBOLS = ["⊕", "⊗", "◇", "▽", "⊘", "⊛", "⊞", "⊠", "△"]
PROBE = (900, 13)
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/app/model.json"))
PUZZLES_PATH = Path(os.environ.get("PUZZLES_PATH", "/app/puzzles.json"))
ANSWER_PATH = Path(os.environ.get("ANSWER_PATH", "/app/answer.json"))

# Hypothesis library: formula (in the lab's model syntax) -> f(a, b, n).
CANDIDATES = {
    "a + b": lambda a, b, n: a + b,
    "a + b + 1": lambda a, b, n: a + b + 1,
    "a + b - 1": lambda a, b, n: a + b - 1,
    "a + b + 2": lambda a, b, n: a + b + 2,
    "a + b + n": lambda a, b, n: a + b + n,
    "(a + b) % 256": lambda a, b, n: (a + b) % 256,
    "a * b": lambda a, b, n: a * b,
    "a * b + 1": lambda a, b, n: a * b + 1,
    "a * b - 1": lambda a, b, n: a * b - 1,
    "a - b": lambda a, b, n: a - b,
    "b - a": lambda a, b, n: b - a,
    "abs(a - b)": lambda a, b, n: abs(a - b),
    "a - b if a >= b else 0": lambda a, b, n: a - b if a >= b else 0,
    "max(a, b)": lambda a, b, n: max(a, b),
    "min(a, b)": lambda a, b, n: min(a, b),
    "a": lambda a, b, n: a,
    "b": lambda a, b, n: b,
    "2*a + b": lambda a, b, n: 2 * a + b,
    "a + 2*b": lambda a, b, n: a + 2 * b,
    "2*a - b": lambda a, b, n: 2 * a - b,
    "2*b - a": lambda a, b, n: 2 * b - a,
    "a + 3*b": lambda a, b, n: a + 3 * b,
    "3*a + b": lambda a, b, n: 3 * a + b,
    "a + b**2": lambda a, b, n: a + b * b,
    "b + a**2": lambda a, b, n: b + a * a,
}


def query(expr: str) -> int:
    out = subprocess.run(["evaluate", expr], capture_output=True, text=True, check=True).stdout
    return int(re.search(r"=\s*(-?\d+)", out).group(1))


# Expression trees: int leaves, (symbol, left, right) nodes.
def render(t):
    if isinstance(t, int):
        return str(t)
    return f"({render(t[1])} {t[0]} {render(t[2])})"


def operands(t):
    return 1 if isinstance(t, int) else operands(t[1]) + operands(t[2])


def symbols_in(t):
    return set() if isinstance(t, int) else {t[0]} | symbols_in(t[1]) | symbols_in(t[2])


def evaluate_under(t, assignment, n):
    if isinstance(t, int):
        return t
    return CANDIDATES[assignment[t[0]]](evaluate_under(t[1], assignment, n), evaluate_under(t[2], assignment, n), n)


def observe(t, hyp):
    """Run one query and keep only hypotheses (jointly, per symbol) that predict its result."""
    value = query(render(t).removeprefix("(").removesuffix(")"))
    syms = sorted(symbols_in(t))
    consistent = [dict(zip(syms, combo)) for combo in itertools.product(*(hyp[s] for s in syms))
                  if evaluate_under(t, dict(zip(syms, combo)), operands(t)) == value]
    for s in syms:
        hyp[s] = sorted({c[s] for c in consistent})


def identify():
    hyp = {s: list(CANDIDATES) for s in SYMBOLS}
    for s in SYMBOLS:                                     # 9 points
        observe((s, *PROBE), hyp)
    observe(("⊠", ("◇", 13, 900), ("▽", 20, 500)), hyp)  # 3 points
    observe(("△", 13, 900), hyp)                         # 1 point
    for s in SYMBOLS:
        print(f"{s}: {hyp[s]}")
    return {s: h[0] for s, h in hyp.items() if len(h) == 1}


def commit_and_reveal(model):
    MODEL_PATH.write_text(json.dumps(model, ensure_ascii=False))
    subprocess.run(["reveal", str(MODEL_PATH)], check=True)
    return json.loads(PUZZLES_PATH.read_text(encoding="utf-8"))


SHAPES = [
    ("{0} {a} {1} {b} {2} {c} {3}", lambda f, n: f(2, f(1, f(0, n[0], n[1]), n[2]), n[3])),
    ("({0} {a} ({1} {b} {2})) {c} {3}", lambda f, n: f(2, f(0, n[0], f(1, n[1], n[2])), n[3])),
    ("({0} {a} {1}) {b} ({2} {c} {3})", lambda f, n: f(1, f(0, n[0], n[1]), f(2, n[2], n[3]))),
    ("{0} {a} (({1} {b} {2}) {c} {3})", lambda f, n: f(0, n[0], f(2, f(1, n[1], n[2]), n[3]))),
    ("{0} {a} ({1} {b} ({2} {c} {3}))", lambda f, n: f(0, n[0], f(1, n[1], f(2, n[2], n[3])))),
]


def solve(model, puzzles):
    answer = {}
    for pid, p in puzzles.items():
        nums, target = p["numbers"], p["target"]
        n = len(nums)
        answer[pid] = "none"
        for ops in itertools.product(SYMBOLS, repeat=3):
            f = lambda i, a, b: CANDIDATES[model[ops[i]]](a, b, n)
            hit = next((fmt for fmt, shape in SHAPES if shape(f, nums) == target), None)
            if hit:
                answer[pid] = hit.format(*nums, a=ops[0], b=ops[1], c=ops[2])
                break
    return answer


if __name__ == "__main__":
    model = identify()
    if set(model) != set(SYMBOLS):
        raise SystemExit("could not identify every operator; refusing to commit a guessed model")
    puzzles = commit_and_reveal(model)
    ans = solve(model, puzzles)
    ANSWER_PATH.write_text(json.dumps(ans, ensure_ascii=False))
    print(json.dumps(ans, ensure_ascii=False))
