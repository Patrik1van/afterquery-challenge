"""Reference solver (the 'oracle').

It knows nothing about the hidden semantics. It works like a careful
experimenter: probe each operator in isolation on two input pairs, match
the observations against a library of candidate operations, then
brute-force every puzzle under the identified semantics.

Uses 8 of the 10 queries. If an operator matches no candidate, the solver
reports it and leaves the affected puzzles unsolved instead of guessing.
"""

import itertools
import json
import os
import re
import subprocess
from pathlib import Path

SYMBOLS = ["⊕", "⊗", "◇", "▽"]

# Probe pairs chosen so that every candidate below gives a distinct signature.
PROBES = [(2, 3), (7, 4)]

CANDIDATES = {
    "a+b": lambda a, b: a + b,
    "a+b+1": lambda a, b: a + b + 1,
    "a+b-1": lambda a, b: a + b - 1,
    "a*b": lambda a, b: a * b,
    "a*b+1": lambda a, b: a * b + 1,
    "a*b-1": lambda a, b: a * b - 1,
    "a-b": lambda a, b: a - b,
    "b-a": lambda a, b: b - a,
    "|a-b|": lambda a, b: abs(a - b),
    "max": lambda a, b: max(a, b),
    "min": lambda a, b: min(a, b),
    "a": lambda a, b: a,
    "b": lambda a, b: b,
}

PUZZLES = {
    "P1": ([5, 3, 8, 2], 74),
    "P2": ([2, 3, 4, 8], 28),
    "P3": ([3, 3, 6, 9], 14),
    "P4": ([7, 8, 3, 5], 16),
    "P5": ([3, 5, 9, 5], 39),
    "P6": ([5, 6, 8, 7], 22),
}


def query(expr: str) -> int:
    out = subprocess.run(["evaluate", expr], capture_output=True, text=True, check=True).stdout
    return int(re.search(r"=\s*(-?\d+)", out).group(1))


def identify():
    semantics = {}
    for sym in SYMBOLS:
        observed = tuple(query(f"{a} {sym} {b}") for a, b in PROBES)
        matches = [name for name, f in CANDIDATES.items()
                   if tuple(f(a, b) for a, b in PROBES) == observed]
        print(f"{sym}: observed {observed} -> {matches or 'no candidate'}")
        if len(matches) == 1:
            semantics[sym] = CANDIDATES[matches[0]]
    return semantics


def solve(semantics):
    answer = {}
    for pid, (nums, target) in PUZZLES.items():
        for combo in itertools.product(semantics, repeat=3):
            acc = nums[0]
            for op, n in zip(combo, nums[1:]):
                acc = semantics[op](acc, n)
            if acc == target:
                answer[pid] = "".join(combo)
                break
    return answer


if __name__ == "__main__":
    sem = identify()
    ans = solve(sem)
    Path(os.environ.get("ANSWER_PATH", "/app/answer.json")).write_text(json.dumps(ans, ensure_ascii=False))
    print(json.dumps(ans, ensure_ascii=False))
