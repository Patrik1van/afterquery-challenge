"""HTTP sidecar that holds the hidden operators, enforces the query budget and reveals the puzzles.

Endpoints:
    POST /evaluate  {"expr": "2 ⊕ (3 ⊘ 1)"}      -> {"result": ..., "cost": 2, "budget_left": ...}
    GET  /budget                                -> {"budget_used": ..., "budget_left": ..., "locked": ...}
    POST /reveal    {"model": {"⊕": "a + b", ...}} -> {"puzzles": {...}, "locked": true}
"""

import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from formula import BadFormula, apply, compile_formula
from generator import generate

BUDGET = int(os.environ.get("LAB_BUDGET", "20"))
QUERY_LOG = os.environ.get("LAB_QUERY_LOG", "/var/log/lab/queries.jsonl")
MODEL_LOG = os.environ.get("LAB_MODEL_LOG", "/var/log/lab/model.json")
PUZZLES_LOG = os.environ.get("LAB_PUZZLES_LOG", "/var/log/lab/puzzles.json")
SEED = int(os.environ.get("FALSIFY_SEED") or 42)
DIFFICULTY = int(os.environ.get("FALSIFY_DIFFICULTY") or 3)
PORT = int(os.environ.get("LAB_PORT", "8000"))

SEMANTICS = json.loads(Path(__file__).with_name("semantics.json").read_text(encoding="utf-8"))
OPS = {sym: compile_formula(spec["rule"]) for sym, spec in SEMANTICS.items()}

MAX_OPERANDS = 4
MAX_OPERAND = 1000
MAX_EXPR_CHARS = 200
TOKEN_RE = re.compile(r"\s*(\d+|[" + "".join(OPS) + "()])")
SYMBOL_LIST = " ".join(OPS)

_lock = threading.Lock()
_used = 0
_locked = False
_puzzles = {}


class BadExpression(ValueError):
    """Raised for a query that is not a valid expression."""


def tokenize(expr):
    """Split a query into numbers, operator symbols and parentheses.

    Args:
        expr: The query string.

    Returns:
        The list of tokens.

    Raises:
        BadExpression: If the query is too long or contains an invalid token.
    """
    if not isinstance(expr, str) or len(expr) > MAX_EXPR_CHARS:
        raise BadExpression(f"expression must be a string of at most {MAX_EXPR_CHARS} characters")
    tokens, pos, expr = [], 0, expr.strip()
    while pos < len(expr):
        m = TOKEN_RE.match(expr, pos)
        if not m:
            raise BadExpression(
                f"invalid token; use non-negative integers, the symbols {SYMBOL_LIST} and parentheses only")
        tokens.append(m.group(1))
        pos = m.end()
    return tokens


def parse(expr):
    """Parse a query into a tree without evaluating it.

    Args:
        expr: The query string; unparenthesized chains are evaluated left to right.

    Returns:
        A tree of int leaves and (symbol, left, right) nodes.

    Raises:
        BadExpression: If the query is malformed or has an invalid number of operands.
    """
    tokens, pos = tokenize(expr), 0

    def peek():
        return tokens[pos] if pos < len(tokens) else None

    def atom():
        nonlocal pos
        tok = peek()
        if tok is None:
            raise BadExpression("unexpected end of expression")
        if tok == "(":
            pos += 1
            node = chain()
            if peek() != ")":
                raise BadExpression("unbalanced parentheses")
            pos += 1
            return node
        if tok.isdigit():
            pos += 1
            value = int(tok)
            if value > MAX_OPERAND:
                raise BadExpression(f"operands must be between 0 and {MAX_OPERAND}")
            return value
        raise BadExpression("expected a number or '('")

    def chain():
        nonlocal pos
        node = atom()
        while peek() in OPS:
            op = tokens[pos]
            pos += 1
            node = (op, node, atom())
        return node

    tree = chain()
    if pos != len(tokens):
        raise BadExpression("expected an operator symbol between operands")
    count = sum(1 for t in tokens if t.isdigit())
    if count < 2:
        raise BadExpression("expected: number op number [op number ...]")
    if count > MAX_OPERANDS:
        raise BadExpression(f"at most {MAX_OPERANDS} operands per expression")
    return tree


def cost(tree):
    """Return the budget cost of a query: one point per operator."""
    if isinstance(tree, int):
        return 0
    _, left, right = tree
    return 1 + cost(left) + cost(right)


def operands(tree):
    """Return the number of operands in a query."""
    return 1 if isinstance(tree, int) else operands(tree[1]) + operands(tree[2])


def compute(tree, n):
    """Evaluate a parsed query under the hidden operators.

    Args:
        tree: The parsed query.
        n: The number of operands in the whole query.

    Returns:
        The integer result.

    Raises:
        ArithmeticError: If an intermediate value is out of range.
    """
    position = 0

    def go(t, depth):
        nonlocal position
        if isinstance(t, int):
            return t
        a = go(t[1], depth + 1)
        position += 1
        i = position
        return apply(OPS[t[0]], a, go(t[2], depth + 1), n, i, depth)
    return go(tree, 0)


def validate_model(model):
    """Check a committed model.

    Args:
        model: Mapping of every operator symbol to a formula string.

    Returns:
        A list of problems; empty if the model is valid.
    """
    if not isinstance(model, dict):
        return ["the model must be a JSON object mapping each operator symbol to a formula"]
    problems = []
    for sym in OPS:
        if sym not in model:
            problems.append(f"missing operator {sym}")
            continue
        try:
            compile_formula(model[sym])
        except BadFormula as e:
            problems.append(f"{sym}: {e}")
    problems += [f"unknown key {k!r}" for k in model if k not in OPS]
    return problems


class Handler(BaseHTTPRequestHandler):
    """Serves /evaluate, /budget and /reveal; requests are handled one at a time."""

    def do_GET(self):
        if self.path == "/budget":
            with _lock:
                self._send(200, self._budget())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path == "/reveal":
            self._reveal()
        elif self.path == "/evaluate":
            self._evaluate()
        else:
            self._send(404, {"error": "not found"})

    def _evaluate(self):
        global _used
        try:
            length = min(int(self.headers.get("Content-Length", 0)), 4096)
            expr = json.loads(self.rfile.read(length) or b"{}")["expr"]
        except Exception:
            self._send(400, {"error": 'body must be JSON like {"expr": "2 ⊕ 3"}'})
            return
        with _lock:
            if _locked:
                self._send(423, {"error": "the lab is locked: the puzzles have been revealed", **self._budget()})
                return
            try:
                tree = parse(expr)
            except BadExpression as e:
                self._send(400, {"error": str(e), **self._budget()})
                return
            price = cost(tree)
            if _used + price > BUDGET:
                self._send(429, {"error": f"query costs {price} but only {BUDGET - _used} budget left",
                                 "cost": price, **self._budget()})
                return
            try:
                result = compute(tree, operands(tree))
            except ArithmeticError:
                result = None
            _used += price
            with open(QUERY_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps({"expr": expr, "result": result, "cost": price, "budget_used": _used},
                                   ensure_ascii=False) + "\n")
            if result is None:
                self._send(400, {"error": "result out of range (charged: the expression was evaluated)",
                                 "cost": price, **self._budget()})
                return
            self._send(200, {"expr": expr, "result": result, "cost": price, **self._budget()})

    def _reveal(self):
        global _locked
        try:
            length = min(int(self.headers.get("Content-Length", 0)), 16384)
            model = json.loads(self.rfile.read(length) or b"{}")["model"]
        except Exception:
            self._send(400, {"error": 'body must be JSON like {"model": {"⊕": "a + b", ...}}'})
            return
        with _lock:
            if _locked:
                self._send(200, {"puzzles": _puzzles, "locked": True, "note": "already revealed; model unchanged"})
                return
            problems = validate_model(model)
            if problems:
                self._send(400, {"error": "model rejected, the lab is still open", "problems": problems})
                return
            with open(MODEL_LOG, "w", encoding="utf-8") as f:
                json.dump({"model": model, "budget_used_before_reveal": _used}, f, ensure_ascii=False)
            _locked = True
            self._send(200, {"puzzles": _puzzles, "locked": True})

    @staticmethod
    def _budget():
        return {"budget_used": _used, "budget_left": BUDGET - _used, "budget": BUDGET, "locked": _locked}

    def _send(self, code, payload):
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        pass


if __name__ == "__main__":
    os.makedirs(os.path.dirname(QUERY_LOG), exist_ok=True)
    open(QUERY_LOG, "a").close()
    with open(MODEL_LOG, "w", encoding="utf-8") as f:
        f.write("null")
    _puzzles = generate(SEMANTICS, SEED, DIFFICULTY)
    Path(PUZZLES_LOG).write_text(json.dumps(_puzzles, ensure_ascii=False))
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
