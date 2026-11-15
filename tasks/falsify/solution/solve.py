"""Reference solver: a generic Bayesian experimenter that knows only what the instruction tells the agent.

Hypotheses are all formulas of the documented grammar up to MAX_SIZE nodes, plus formulas
with one larger constant (short-to-describe constants up front, any other value fitted from
an observation). The prior favours short formulas and penalises the context variables n, i, d.
Each round runs the query with the highest expected information per budget point and removes
contradicted hypotheses. The most probable formula per operator is committed, and the puzzles
are brute-forced under that model.
"""

import itertools
import json
import math
import os
import random
import re
import subprocess
import sys
import time
from pathlib import Path

SYMBOLS = ["⊕", "⊗", "◇", "▽", "⊘", "⊛", "⊞", "⊠", "△"]
LIMIT = 10 ** 15
MAX_SIZE = int(os.environ.get("SOLVER_MAX_SIZE", 6))
LAMBDA = float(os.environ.get("SOLVER_LAMBDA", 2.0))
TAU = float(os.environ.get("SOLVER_TAU", 0.95))
CTX_PENALTY = float(os.environ.get("SOLVER_CTX_PENALTY", 1.0))
POOL = [0, 1, 2, 3, 5, 7, 10, 13, 20, 50, 100, 300, 500, 700, 900, 1000]
SMALL = [1, 2, 3, 5, 7, 13, 20, 100, 500, 900]
TOP_M = 1500
TOP_JOINT = 6
TOP_UPDATE = 30
N_MULTI = 1500
LOG = os.environ.get("SOLVER_QUIET") is None
NICE = sorted({*range(3, 11), *(2 ** k for k in range(4, 13)), *(2 ** k - 1 for k in range(4, 13)), 100, 1000, 99, 999})
BINOPS = [("+", True), ("-", False), ("*", True), ("//", False), ("%", False), ("**", False), ("max", True), ("min", True)]
CTX_RE = re.compile(r"\b[nid]\b")


def log(*args):
    """Print progress to stderr unless SOLVER_QUIET is set."""
    if LOG:
        print(*args, file=sys.stderr, flush=True)


def P(x, y):
    """Return x ** y with the lab's limits on the exponent and base."""
    if not 0 <= y <= 12 or abs(x) > LIMIT:
        raise ArithmeticError
    return x ** y


def guard(v):
    """Return v as an int if it is within the lab's value range, else None."""
    if isinstance(v, bool):
        v = int(v)
    return v if isinstance(v, int) and -LIMIT <= v <= LIMIT else None


def compile_py(py):
    """Compile a Python-syntax formula into f(a, b, n, i, d) that returns None on an error."""
    fn = eval(f"lambda a, b, n, i, d: {py}", {"P": P, "max": max, "min": min, "abs": abs})

    def f(a, b, n, i, d):
        try:
            return guard(fn(a, b, n, i, d))
        except (ArithmeticError, TypeError, ValueError, OverflowError):
            return None
    return f


class Hyp:
    """A candidate formula for an operator, with its display form, Python form and prior cost."""

    __slots__ = ("src", "py", "cost", "_f")

    def __init__(self, src, py, cost):
        """Create a hypothesis.

        Args:
            src: The formula in the lab's grammar, as committed to model.json.
            py: The same formula in Python syntax with P() for powers.
            cost: Size-based cost; each use of n, i or d adds CTX_PENALTY.
        """
        self.src, self.py, self._f = src, py, None
        self.cost = cost + CTX_PENALTY * len(CTX_RE.findall(src))

    def f(self, a, b, n, i, d):
        """Evaluate the formula, compiling it on first use."""
        if self._f is None:
            self._f = compile_py(self.py)
        return self._f(a, b, n, i, d)


def dedupe_points():
    """Return the fixed (a, b, n, i, d) points used to merge formulas that behave identically."""
    rng = random.Random(0)
    vals = [-50, -3, 0, 1, 2, 3, 5, 7, 13, 20, 100, 255, 300, 513, 900, 1000, 2000, 99999]
    ctx = [(2, 1, 0)] * 40 + [(n, i, d) for n in (3, 4) for i in range(1, n) for d in range(0, n - 1)] * 4
    return [(rng.choice(vals), rng.choice(vals), *c) for c in ctx]


D0 = dedupe_points()


def vec_bin(op, xs, ys):
    """Apply a binary operator element-wise to two value vectors; errors become None."""
    out = []
    for x, y in zip(xs, ys):
        if x is None or y is None:
            out.append(None)
            continue
        try:
            if op == "+":
                v = x + y
            elif op == "-":
                v = x - y
            elif op == "*":
                v = x * y
            elif op == "//":
                v = x // y
            elif op == "%":
                v = x % y
            elif op == "**":
                v = P(x, y)
            elif op == "max":
                v = max(x, y)
            else:
                v = min(x, y)
            out.append(guard(v))
        except ArithmeticError:
            out.append(None)
    return tuple(out)


def fmt(op, left, right):
    """Return the (lab grammar, Python) source of a binary node from its operands' sources."""
    if op in ("max", "min"):
        return f"{op}({left[0]}, {right[0]})", f"{op}({left[1]}, {right[1]})"
    if op == "**":
        return f"({left[0]} ** {right[0]})", f"P({left[1]}, {right[1]})"
    return f"({left[0]} {op} {right[0]})", f"({left[1]} {op} {right[1]})"


def const_cost(c):
    """Return the prior cost of a constant: short-to-describe values are cheaper."""
    c = abs(c)
    if c <= 2:
        return 1.0
    nice = c <= 10 or any(c in (10 ** k, 2 ** k, 2 ** k - 1, 10 ** k - 1) for k in range(2, 21))
    return 2.0 if nice else 2.0 + 0.5 * math.log2(c)


def enumerate_space():
    """Enumerate the hypothesis space bottom-up, keeping one formula per distinct behaviour on D0.

    Returns:
        A (hypotheses, hashes of seen behaviours, small expressions of size <= 3) tuple; the
        small expressions are (vector, src, py, size) tuples used for fitting constants.
    """
    t0 = time.time()
    seen, hyps, vecs = set(), [], {}

    def add(vec, src, py, size, cost=None, keep_vec=True):
        if all(v is None for v in vec):
            return
        h = hash(vec)
        if h in seen:
            return
        seen.add(h)
        hyps.append(Hyp(src, py, size if cost is None else cost))
        if keep_vec:
            vecs.setdefault(size, []).append((vec, src, py))

    for k, name in enumerate("abnid"):
        add(tuple(p[k] for p in D0), name, name, 1)
    for c in (0, 1, 2):
        add(tuple(c for _ in D0), str(c), str(c), 1)
    for s in range(2, MAX_SIZE + 1):
        keep = s < MAX_SIZE
        for vec, src, py in vecs.get(s - 1, []):
            add(tuple(None if v is None else abs(v) for v in vec), f"abs({src})", f"abs({py})", s, keep_vec=keep)
            add(tuple(None if v is None else -v for v in vec), f"-({src})", f"-({py})", s, keep_vec=keep)
        for ls in range(1, s - 1):
            rs = s - 1 - ls
            for op, comm in BINOPS:
                for lv, lsrc, lpy in vecs.get(ls, []):
                    for rv, rsrc, rpy in vecs.get(rs, []):
                        if comm and (ls > rs or (ls == rs and lsrc > rsrc)):
                            continue
                        src, py = fmt(op, (lsrc, lpy), (rsrc, rpy))
                        add(vec_bin(op, lv, rv), src, py, s, keep_vec=keep)
        if s >= 6:
            leaves = vecs.get(1, [])
            for pi, (pv, psrc, _) in enumerate(leaves):
                for qv, qsrc, _ in leaves[pi + 1:]:
                    cond = [None if p is None or q is None else p >= q for p, q in zip(pv, qv)]
                    for bs in range(1, s - 4):
                        for xv, xsrc, xpy in vecs.get(bs, []):
                            for yv, ysrc, ypy in vecs.get(s - 4 - bs, []):
                                vec = tuple(None if c is None else (x if c else y) for c, x, y in zip(cond, xv, yv))
                                add(vec, f"({xsrc} if {psrc} >= {qsrc} else {ysrc})",
                                    f"({xpy} if {psrc} >= {qsrc} else {ypy})", s, keep_vec=keep)
    n_plain = len(hyps)
    small = [(*v, s) for s in (1, 2, 3) for v in vecs.get(s, [])]
    for vec, src, py, size in small:
        for c in NICE:
            cv = tuple(c for _ in D0)
            for op in ("+", "-", "*", "//", "%", "max", "min"):
                s2, p2 = fmt(op, (src, py), (str(c), str(c)))
                add(vec_bin(op, vec, cv), s2, p2, None, cost=size + 1 + const_cost(c), keep_vec=False)
            s2, p2 = fmt("-", (str(c), str(c)), (src, py))
            add(vec_bin("-", cv, vec), s2, p2, None, cost=size + 1 + const_cost(c), keep_vec=False)
    log(f"hypotheses: {n_plain} formulas up to size {MAX_SIZE} + {len(hyps) - n_plain} with a constant "
        f"({time.time() - t0:.1f}s)")
    return hyps, seen, small


def render(t, top=True):
    """Render a query tree (int leaves, (symbol, left, right) nodes) as an expression string."""
    if isinstance(t, int):
        return str(t)
    s = f"{render(t[1], False)} {t[0]} {render(t[2], False)}"
    return s if top else f"({s})"


def symbols_in(t):
    """Return the operator symbols of a query tree in order."""
    return [] if isinstance(t, int) else [t[0]] + symbols_in(t[1]) + symbols_in(t[2])


def n_operands(t):
    """Return the number of operands in a query tree."""
    return 1 if isinstance(t, int) else n_operands(t[1]) + n_operands(t[2])


def evaluate(t, assign):
    """Evaluate a query tree with each symbol read as its assigned hypothesis, using the lab's n, i, d."""
    n, pos = n_operands(t), 0

    def go(node, depth):
        nonlocal pos
        if isinstance(node, int):
            return node
        a = go(node[1], depth + 1)
        pos += 1
        i = pos
        b = go(node[2], depth + 1)
        if a is None or b is None:
            return None
        return assign[node[0]].f(a, b, n, i, depth)
    return go(t, 0)


SHAPES = {
    3: [lambda s, x: (s[1], (s[0], x[0], x[1]), x[2]), lambda s, x: (s[0], x[0], (s[1], x[1], x[2]))],
    4: [lambda s, x: (s[2], (s[1], (s[0], x[0], x[1]), x[2]), x[3]),
        lambda s, x: (s[2], (s[0], x[0], (s[1], x[1], x[2])), x[3]),
        lambda s, x: (s[1], (s[0], x[0], x[1]), (s[2], x[2], x[3])),
        lambda s, x: (s[0], x[0], (s[2], (s[1], x[1], x[2]), x[3])),
        lambda s, x: (s[0], x[0], (s[1], x[1], (s[2], x[2], x[3])))],
}


def entropy(weights_by_outcome):
    """Return the Shannon entropy in bits of a weighted outcome distribution."""
    tot = sum(weights_by_outcome.values())
    if tot <= 0:
        return 0.0
    return -sum(w / tot * math.log2(w / tot) for w in weights_by_outcome.values() if w > 0)


class Solver:
    """Chooses queries, updates the hypotheses per operator and returns the most probable model."""

    def __init__(self, lab):
        """Build the hypothesis space.

        Args:
            lab: An object with query(expr) -> int or None and budget_left() -> int.
        """
        self.lab = lab
        self.hyps, self.seen, self.small = enumerate_space()
        self.alive = {s: list(range(len(self.hyps))) for s in SYMBOLS}
        self.obs = []
        self.cache = {}
        self.rng = random.Random(1)

    def w(self, k):
        """Return the prior weight of hypothesis k."""
        return math.exp(-LAMBDA * self.hyps[k].cost)

    def ranked(self, s, m):
        """Return the m most probable live hypotheses for symbol s."""
        return sorted(self.alive[s], key=lambda k: self.hyps[k].cost)[:m]

    def share(self, s):
        """Return the posterior share of the most probable hypothesis for symbol s."""
        ws = [self.w(k) for k in self.alive[s]]
        return max(ws) / sum(ws) if ws else 0.0

    def single(self, k, a, b):
        """Return hypothesis k's prediction for a two-operand query, cached."""
        key = (k, a, b)
        if key not in self.cache:
            self.cache[key] = self.hyps[k].f(a, b, 2, 1, 0)
        return self.cache[key]

    def best_query(self, budget_left):
        """Find the affordable query with the highest expected information per budget point.

        Args:
            budget_left: Points still available.

        Returns:
            A (query tree, bits per point) tuple, or (None, 0.0) if no query is informative.
        """
        best = (0.0, None, 0)
        for s in SYMBOLS:
            if len(self.alive[s]) < 2:
                continue
            top = self.ranked(s, TOP_M)
            ws = [self.w(k) for k in top]
            for a in POOL:
                for b in POOL:
                    buckets = {}
                    for k, wk in zip(top, ws):
                        v = self.single(k, a, b)
                        buckets[v] = buckets.get(v, 0.0) + wk
                    h = entropy(buckets)
                    if h > best[0] + 1e-9:
                        best = (h, (s, a, b), 1)
        open_syms = [s for s in SYMBOLS if self.share(s) < TAU] or SYMBOLS
        for _ in range(N_MULTI):
            n = self.rng.choice([3, 4])
            if n - 1 > budget_left:
                continue
            syms = [self.rng.choice(open_syms if self.rng.random() < 0.7 else SYMBOLS) for _ in range(n - 1)]
            tree = self.rng.choice(SHAPES[n])(syms, [self.rng.choice(SMALL) for _ in range(n)])
            h = self.score_joint(tree)
            if h / (n - 1) > (best[0] / best[2] if best[1] else 0.0) + 1e-9:
                best = (h, tree, n - 1)
        if best[1] is None:
            return None, 0.0
        return best[1], best[0] / best[2]

    def score_joint(self, tree):
        """Return the entropy of a multi-operator query's outcome under the top joint hypotheses."""
        syms = sorted(set(symbols_in(tree)))
        tops = [self.ranked(s, TOP_JOINT) for s in syms]
        buckets = {}
        for combo in itertools.product(*tops):
            assign = {s: self.hyps[k] for s, k in zip(syms, combo)}
            wk = math.prod(self.w(k) for k in combo)
            v = evaluate(tree, assign)
            buckets[v] = buckets.get(v, 0.0) + wk
        return entropy(buckets)

    def observe(self, tree, value):
        """Remove the hypotheses a query result contradicts.

        Args:
            tree: The query that was run.
            value: Its result, or None if the lab reported an error.
        """
        self.obs.append((tree, value))
        syms = sorted(set(symbols_in(tree)))
        if len(syms) == 1 and len(symbols_in(tree)) == 1:
            s, a, b = tree
            before = len(self.alive[s])
            self.alive[s] = [k for k in self.alive[s] if self.single(k, a, b) == value]
            self.fit_constants(s, a, b, value)
            log(f"  {s}: {before} -> {len(self.alive[s])} hypotheses")
            return
        for s in syms:
            others = [o for o in syms if o != s]
            tops = [self.ranked(o, TOP_UPDATE) for o in others]
            keep = []
            for k in self.alive[s]:
                for combo in itertools.product(*tops):
                    assign = {o: self.hyps[c] for o, c in zip(others, combo)}
                    assign[s] = self.hyps[k]
                    if evaluate(tree, assign) == value:
                        keep.append(k)
                        break
            before = len(self.alive[s])
            if keep:
                self.alive[s] = keep
            log(f"  {s}: {before} -> {len(self.alive[s])} hypotheses")

    def fit_constants(self, s, a, b, v):
        """Add "E op C" hypotheses for symbol s whose constant C is determined by the result a s b = v."""
        if v is None:
            return
        new = 0
        for _, src, py, size in self.small:
            e = compile_py(py)(a, b, 2, 1, 0)
            if e is None:
                continue
            cands = {("+", v - e), ("-", e - v), ("C-", v + e)}
            if e and v % e == 0:
                cands.add(("*", v // e))
            if e > v >= 0 and e - v <= 10 ** 12:
                diff, c = e - v, 1
                while c * c <= diff:
                    if diff % c == 0:
                        cands |= {("%", x) for x in (c, diff // c) if x > v}
                    c += 1
            for op, c in cands:
                if abs(c) <= 2 or abs(c) > 10 ** 6:
                    continue
                if op == "C-":
                    s2, p2 = fmt("-", (str(c), str(c)), (src, py))
                else:
                    s2, p2 = fmt(op, (src, py), (str(c), str(c)))
                h = Hyp(s2, p2, size + 1 + const_cost(c))
                vec2 = tuple(h.f(*p) for p in D0)
                if all(x is None for x in vec2) or hash(vec2) in self.seen:
                    continue
                if not self.consistent(s, h):
                    continue
                self.seen.add(hash(vec2))
                self.hyps.append(h)
                self.alive[s].append(len(self.hyps) - 1)
                new += 1
        if new:
            log(f"  {s}: +{new} hypotheses with a fitted constant")

    def consistent(self, s, h):
        """Return whether hypothesis h for symbol s agrees with every observation so far."""
        for tree, value in self.obs:
            syms = symbols_in(tree)
            if s not in syms:
                continue
            if len(syms) == 1:
                if h.f(tree[1], tree[2], 2, 1, 0) != value:
                    return False
                continue
            others = sorted(set(syms) - {s})
            ok = False
            for combo in itertools.product(*[self.ranked(o, TOP_UPDATE) for o in others]):
                assign = {o: self.hyps[c] for o, c in zip(others, combo)}
                assign[s] = h
                if evaluate(tree, assign) == value:
                    ok = True
                    break
            if not ok:
                return False
        return True

    def run(self):
        """Experiment until every operator is resolved or the budget is spent.

        Returns:
            Mapping of symbol to its most probable hypothesis.
        """
        while True:
            left = self.lab.budget_left()
            shares = {s: self.share(s) for s in SYMBOLS}
            log("budget left", left, " ".join(f"{s}:{self.map(s).src}({shares[s]:.2f})" for s in SYMBOLS))
            if left <= 0 or all(v >= TAU for v in shares.values()):
                break
            tree, gain = self.best_query(left)
            if tree is None or gain <= 1e-6:
                break
            value = self.lab.query(render(tree))
            log(f"query {render(tree)} = {value}   (expected gain {gain:.2f} bits/point)")
            self.observe(tree, value)
        return {s: self.map(s) for s in SYMBOLS}

    def map(self, s):
        """Return the most probable live hypothesis for symbol s."""
        return self.hyps[min(self.alive[s], key=lambda k: self.hyps[k].cost)]


def solve_puzzles(hyps, puzzles):
    """Brute-force each puzzle under the committed model.

    Args:
        hyps: Mapping of symbol to its committed hypothesis.
        puzzles: Mapping of puzzle id to {"numbers": [...], "target": t}.

    Returns:
        Mapping of puzzle id to an expression that reaches the target, or "none".
    """
    answer = {}
    for pid, p in puzzles.items():
        nums, target = p["numbers"], p["target"]
        answer[pid] = "none"
        for ops in itertools.product(SYMBOLS, repeat=3):
            hit = next((t for t in (shape(list(ops), nums) for shape in SHAPES[4]) if evaluate(t, hyps) == target), None)
            if hit:
                answer[pid] = render(hit)
                break
    return answer


class RealLab:
    """Access to the lab through the evaluate command inside the task container."""

    def query(self, expr):
        """Run a query and return its result, or None if the lab reported an error."""
        r = subprocess.run(["evaluate", expr], capture_output=True, text=True)
        m = re.search(r"=\s*(-?\d+)\s+\(cost", r.stdout)
        return int(m.group(1)) if m else None

    def budget_left(self):
        """Return the remaining budget."""
        out = subprocess.run(["evaluate", "--budget"], capture_output=True, text=True).stdout
        return int(re.search(r"left:\s*(\d+)", out).group(1))


def main():
    """Experiment, commit the model, reveal the puzzles and write the answers."""
    model_path = Path(os.environ.get("MODEL_PATH", "/app/model.json"))
    puzzles_path = Path(os.environ.get("PUZZLES_PATH", "/app/puzzles.json"))
    answer_path = Path(os.environ.get("ANSWER_PATH", "/app/answer.json"))
    hyps = Solver(RealLab()).run()
    model = {s: h.src for s, h in hyps.items()}
    log("model", model)
    model_path.write_text(json.dumps(model, ensure_ascii=False))
    subprocess.run(["reveal", str(model_path)], check=True)
    puzzles = json.loads(puzzles_path.read_text(encoding="utf-8"))
    answer = solve_puzzles(hyps, puzzles)
    answer_path.write_text(json.dumps(answer, ensure_ascii=False))
    print(json.dumps(answer, ensure_ascii=False))


if __name__ == "__main__":
    main()
