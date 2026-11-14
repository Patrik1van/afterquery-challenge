"""The 'lab': a sidecar that holds the hidden operator semantics.

The agent's container can only reach this service over HTTP. It never sees
this file, so the meaning of the operators cannot be read, only probed.

Endpoints
  POST /evaluate  {"expr": "2 ⊕ (3 ⊘ 1)"}  -> {"result": 8, "queries_used": 1, "queries_left": 7}
  GET  /budget                            -> {"queries_used": 1, "queries_left": 7, "budget": 8}

Rules
  * Without parentheses, evaluation is strictly left to right: a ? b ? c = (a ? b) ? c.
  * Parentheses group sub-expressions and may be nested.
  * Operands are integers 0..1000, at most 4 operands per expression.
  * The whole expression is validated before anything is computed, so
    malformed expressions return an error for free and reveal nothing.
  * After BUDGET successful evaluations every request is refused.

Every successful query is appended to QUERY_LOG, which the verifier collects
as an artifact (the agent cannot write to it).
"""

import json
import os
import re
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

BUDGET = int(os.environ.get("LAB_BUDGET", "8"))
QUERY_LOG = os.environ.get("LAB_QUERY_LOG", "/var/log/lab/queries.jsonl")
PORT = int(os.environ.get("LAB_PORT", "8000"))

# ---- hidden semantics (v1: ⊘ and ⊛ are not commutative) ----
OPS = {
    "⊕": lambda a, b: a + b + 1,
    "⊗": lambda a, b: a * b - 1,
    "◇": lambda a, b: max(a, b),
    "▽": lambda a, b: abs(a - b),
    "⊘": lambda a, b: 2 * a - b,
    "⊛": lambda a, b: a + b * b,
}

MAX_OPERANDS = 4
MAX_OPERAND = 1000
MAX_EXPR_CHARS = 200
TOKEN_RE = re.compile(r"\s*(\d+|[⊕⊗◇▽⊘⊛()])")
SYMBOL_LIST = " ".join(OPS)

_lock = threading.Lock()
_used = 0


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


def compute(tree):
    if isinstance(tree, int):
        return tree
    op, left, right = tree
    return OPS[op](compute(left), compute(right))


def evaluate(expr: str) -> int:
    return compute(parse(expr))


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/budget":
            with _lock:
                self._send(200, self._budget())
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        global _used
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
            if _used >= BUDGET:
                self._send(429, {"error": "query budget exhausted", **self._budget()})
                return
            try:
                result = evaluate(expr)
            except BadExpression as e:
                self._send(400, {"error": str(e), **self._budget()})
                return
            _used += 1
            with open(QUERY_LOG, "a", encoding="utf-8") as f:
                f.write(json.dumps({"n": _used, "expr": expr, "result": result}, ensure_ascii=False) + "\n")
            self._send(200, {"expr": expr, "result": result, **self._budget()})

    @staticmethod
    def _budget():
        return {"queries_used": _used, "queries_left": BUDGET - _used, "budget": BUDGET}

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
    open(QUERY_LOG, "a").close()  # exists even if the agent never queries
    # Single-threaded server: requests are handled one at a time, so the
    # budget cannot be raced with parallel requests.
    HTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
