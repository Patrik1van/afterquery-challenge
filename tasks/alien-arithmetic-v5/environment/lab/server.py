"""The 'lab': a sidecar that holds the hidden operator semantics.

The agent's container can only reach this service over HTTP. It never sees
this file, so the meaning of the operators cannot be read, only probed.

Two phases
  1. Experiment: the agent may query /evaluate. It does not know the puzzles.
  2. Puzzles: the agent commits its model of every operator via /reveal. The
     lab validates the model, stores it, locks itself (no more /evaluate), and
     only then returns the puzzles. The puzzles never exist in the agent's
     container before this point.

Endpoints
  POST /evaluate  {"expr": "2 ⊕ (3 ⊘ 1)"}  -> {"result": 8, "cost": 2, "budget_used": 2, "budget_left": 6, "budget": 8}
  GET  /budget                            -> {"budget_used": 2, "budget_left": 6, "budget": 8, "locked": false}
  POST /reveal    {"model": {"⊕": "a + b", ...}}  -> {"puzzles": {...}, "locked": true}

Rules
  * Without parentheses, evaluation is strictly left to right: a ? b ? c = (a ? b) ? c.
  * Parentheses group sub-expressions and may be nested.
  * Operands are integers 0..1000, at most 4 operands per expression.
  * The whole expression is validated before anything is computed, so
    malformed expressions return an error for free and reveal nothing.
  * A successful query costs one budget point per operator in it:
    "a ? b" costs 1, "a ? b ? c" costs 2, "(a ? b) ? (c ? d)" costs 3.
    Combining probes therefore saves nothing.
  * A query that costs more than the remaining budget is refused for free,
    before anything is computed.

Every successful query is appended to QUERY_LOG and the committed model is
written to MODEL_LOG. The verifier collects both as artifacts (the agent
cannot write to them).
"""

import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from formula import BadFormula, compile_formula

BUDGET = int(os.environ.get("LAB_BUDGET", "13"))
QUERY_LOG = os.environ.get("LAB_QUERY_LOG", "/var/log/lab/queries.jsonl")
MODEL_LOG = os.environ.get("LAB_MODEL_LOG", "/var/log/lab/model.json")
PUZZLES_PATH = os.environ.get("LAB_PUZZLES", str(Path(__file__).with_name("puzzles.json")))
PORT = int(os.environ.get("LAB_PORT", "8000"))

# ---- hidden semantics ----
# Every operator takes (a, b, n), where n is the number of operands in the whole
# expression being evaluated. Only ⊠ uses n: "a ⊠ b" gives a+b+2, but inside a
# 4-operand expression every ⊠ adds 4.
OPS = {
    "⊕": lambda a, b, n: a + b + 1,
    "⊗": lambda a, b, n: a * b - 1,
    "◇": lambda a, b, n: max(a, b),
    "▽": lambda a, b, n: abs(a - b),
    "⊘": lambda a, b, n: 2 * a - b,
    "⊛": lambda a, b, n: a + b * b,
    "⊞": lambda a, b, n: (a + b) % 256,
    "⊠": lambda a, b, n: a + b + n,
    "△": lambda a, b, n: a - b if a >= b else 0,
}

MAX_OPERANDS = 4
MAX_OPERAND = 1000
MAX_EXPR_CHARS = 200
TOKEN_RE = re.compile(r"\s*(\d+|[⊕⊗◇▽⊘⊛⊞⊠△()])")
SYMBOL_LIST = " ".join(OPS)

_lock = threading.Lock()
_used = 0
_locked = False
_puzzles = {}


class BadExpression(ValueError):
    pass


def tokenize(expr):
    if not isinstance(expr, str) or len(expr) > MAX_EXPR_CHARS:
        raise BadExpression(f"expression must be a string of at most {MAX_EXPR_CHARS} characters")
    tokens, pos, expr = [], 0, expr.strip()
    while pos < len(expr):
        m = TOKEN_RE.match(expr, pos)
        if not m:
            raise BadExpression(
                f"invalid token; use non-negative integers, the symbols {SYMBOL_LIST} "
                "and parentheses only"
            )
        tokens.append(m.group(1))
        pos = m.end()
    return tokens


def parse(expr):
    """Parse into a tree: int leaves, (op, left, right) nodes. Never evaluates."""
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
    operands = sum(1 for t in tokens if t.isdigit())
    if operands < 2:
        raise BadExpression("expected: number op number [op number ...]")
    if operands > MAX_OPERANDS:
        raise BadExpression(f"at most {MAX_OPERANDS} operands per expression")
    return tree


def cost(tree):
    """Budget points for a query: one per operator."""
    if isinstance(tree, int):
        return 0
    _, left, right = tree
    return 1 + cost(left) + cost(right)


def operands(tree):
    return 1 if isinstance(tree, int) else operands(tree[1]) + operands(tree[2])


def compute(tree, n):
    if isinstance(tree, int):
        return tree
    op, left, right = tree
    return OPS[op](compute(left, n), compute(right, n), n)




class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/budget":
            with _lock:
                self._send(200, self._budget())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        global _used
        if self.path == "/reveal":
            self._reveal()
            return
        if self.path != "/evaluate":
            self._send(404, {"error": "not found"})
            return
        try:
            length = min(int(self.headers.get("Content-Length", 0)), 4096)
            body = json.loads(self.rfile.read(length) or b"{}")
            expr = body["expr"]
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
            result = compute(tree, operands(tree))
            _used += price
            with open(QUERY_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps({"expr": expr, "result": result, "cost": price, "budget_used": _used},
                                   ensure_ascii=False) + "\n")
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
                # Idempotent: show the puzzles again, the committed model stays as it was.
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


def validate_model(model):
    """Return a list of problems; empty means the model is acceptable."""
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


if __name__ == "__main__":
    os.makedirs(os.path.dirname(QUERY_LOG), exist_ok=True)
    open(QUERY_LOG, "a").close()  # exists even if the agent never queries
    with open(MODEL_LOG, "w", encoding="utf-8") as f:
        f.write("null")  # stays null if the agent never reveals
    _puzzles = json.loads(Path(PUZZLES_PATH).read_text(encoding="utf-8"))
    # Single-threaded server: requests are handled one at a time, so the
    # budget cannot be raced with parallel requests.
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
