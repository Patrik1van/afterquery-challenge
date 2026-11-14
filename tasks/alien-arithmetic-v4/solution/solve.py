"""Reference solver (the 'oracle') for the two-phase task.

Phase 1. It knows nothing about the hidden semantics and nothing about the
puzzles. It probes every operator once on (100, 7), a pair where almost every
candidate gives a distinct value, and re-probes with the operands swapped only
when the first answer is still ambiguous (◇ and ▽ here). 6 + 2 = 8 points.

Commit. It writes the identified formulas to /app/model.json and runs
`reveal`, which locks the lab and returns the puzzles.

Phase 2. It brute-forces every puzzle over all operator choices and all five
ways to parenthesize four operands, answering "none" when nothing fits.
"""

import itertools
import json
import os
import re
import subprocess
from pathlib import Path

SYMBOLS = ["⊕", "⊗", "◇", "▽", "⊘", "⊛"]
FIRST_PROBE, SECOND_PROBE = (100, 7), (7, 100)
MODEL_PATH = Path(os.environ.get("MODEL_PATH", "/app/model.json"))
PUZZLES_PATH = Path(os.environ.get("PUZZLES_PATH", "/app/puzzles.json"))
ANSWER_PATH = Path(os.environ.get("ANSWER_PATH", "/app/answer.json"))

# candidate name (a formula in the lab's model syntax) -> implementation
CANDIDATES = {
    "a + b": lambda a, b: a + b,
    "a + b + 1": lambda a, b: a + b + 1,
    "a + b - 1": lambda a, b: a + b - 1,
    "a * b": lambda a, b: a * b,
    "a * b + 1": lambda a, b: a * b + 1,
    "a * b - 1": lambda a, b: a * b - 1,
    "a - b": lambda a, b: a - b,
    "b - a": lambda a, b: b - a,
    "abs(a - b)": lambda a, b: abs(a - b),
    "max(a, b)": lambda a, b: max(a, b),
    "min(a, b)": lambda a, b: min(a, b),
    "a": lambda a, b: a,
    "b": lambda a, b: b,
    "2*a + b": lambda a, b: 2 * a + b,
    "a + 2*b": lambda a, b: a + 2 * b,
    "2*a - b": lambda a, b: 2 * a - b,
    "2*b - a": lambda a, b: 2 * b - a,
    "a + 3*b": lambda a, b: a + 3 * b,
    "3*a + b": lambda a, b: 3 * a + b,
    "a + b**2": lambda a, b: a + b * b,
    "b + a**2": lambda a, b: b + a * a,
}

# The five ways to parenthesize n0 ? n1 ? n2 ? n3 (shape 0 is left to right).
SHAPES = [
    ("{0} {a} {1} {b} {2} {c} {3}", lambda f, n: f(2, f(1, f(0, n[0], n[1]), n[2]), n[3])),
    ("({0} {a} ({1} {b} {2})) {c} {3}", lambda f, n: f(2, f(0, n[0], f(1, n[1], n[2])), n[3])),
    ("({0} {a} {1}) {b} ({2} {c} {3})", lambda f, n: f(1, f(0, n[0], n[1]), f(2, n[2], n[3]))),
    ("{0} {a} (({1} {b} {2}) {c} {3})", lambda f, n: f(0, n[0], f(2, f(1, n[1], n[2]), n[3]))),
    ("{0} {a} ({1} {b} ({2} {c} {3}))", lambda f, n: f(0, n[0], f(1, n[1], f(2, n[2], n[3])))),
]


def query(expr: str) -> int:
    out = subprocess.run(["evaluate", expr], capture_output=True, text=True, check=True).stdout
    return int(re.search(r"=\s*(-?\d+)", out).group(1))


def matching(probe, value, names):
    return [n for n in names if CANDIDATES[n](*probe) == value]


def identify():
    """Return {symbol: formula} for every operator identified unambiguously."""
    model = {}
    for sym in SYMBOLS:
        matches = matching(FIRST_PROBE, query("{} {} {}".format(FIRST_PROBE[0], sym, FIRST_PROBE[1])), CANDIDATES)
        if len(matches) > 1:
            value = query("{} {} {}".format(SECOND_PROBE[0], sym, SECOND_PROBE[1]))
            matches = matching(SECOND_PROBE, value, matches)
        print(f"{sym}: {matches or 'no candidate'}")
        if len(matches) == 1:
            model[sym] = matches[0]
    return model


def commit_and_reveal(model):
    MODEL_PATH.write_text(json.dumps(model, ensure_ascii=False))
    subprocess.run(["reveal", str(MODEL_PATH)], check=True)
    return json.loads(PUZZLES_PATH.read_text(encoding="utf-8"))


def solve(model, puzzles):
    semantics = {sym: CANDIDATES[formula] for sym, formula in model.items()}
    answer = {}
    for pid, p in puzzles.items():
        nums, target = p["numbers"], p["target"]
        answer[pid] = "none"
        for ops in itertools.product(semantics, repeat=3):
            f = lambda i, a, b: semantics[ops[i]](a, b)
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
