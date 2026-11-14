"""Safe parser and evaluator for operator formulas such as "a + b**2" or "max(a, b)".

Used by the lab (to validate a committed model before locking) and by the
verifier (to score it). The two copies, environment/lab/formula.py and
tests/formula.py, must stay identical.

Formulas are parsed with `ast` against a whitelist and are never passed to
eval(). Allowed: integer literals, the operands a and b, three context
variables (n = number of operands in the whole expression, i = position of
this operator counted from the left starting at 1, d = nesting depth with 0
for the operation evaluated last), + - * // % **,
unary minus, abs/max/min, comparisons, and/or/not, and "x if cond else y".
"""

import ast

FUNCS = {"abs": abs, "max": max, "min": min}
BINOPS = {
    ast.Add: lambda x, y: x + y,
    ast.Sub: lambda x, y: x - y,
    ast.Mult: lambda x, y: x * y,
    ast.FloorDiv: lambda x, y: x // y,
    ast.Mod: lambda x, y: x % y,
    ast.Pow: lambda x, y: x ** y,
}
CMPOPS = {
    ast.Lt: lambda x, y: x < y, ast.LtE: lambda x, y: x <= y,
    ast.Gt: lambda x, y: x > y, ast.GtE: lambda x, y: x >= y,
    ast.Eq: lambda x, y: x == y, ast.NotEq: lambda x, y: x != y,
}
MAX_LEN = 200
MAX_ABS = 10 ** 15
MAX_EXPONENT = 12
SYNTAX_HELP = ("use integers, a, b, n, i, d, + - * // % **, abs/max/min, comparisons, "
               "and/or/not, and 'x if cond else y'")


class BadFormula(ValueError):
    pass


def compile_formula(src):
    """Return a checked AST for a formula string, or raise BadFormula."""
    if not isinstance(src, str) or not src.strip() or len(src) > MAX_LEN:
        raise BadFormula(f"a formula must be a non-empty string of at most {MAX_LEN} characters")
    try:
        tree = ast.parse(src.strip(), mode="eval")
    except SyntaxError as e:
        raise BadFormula(f"not a valid expression ({e.msg})") from None
    _check(tree.body)
    return tree.body


def _check(node):
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return
    if isinstance(node, ast.Name) and node.id in ("a", "b", "n", "i", "d"):
        return
    if isinstance(node, ast.BinOp) and type(node.op) in BINOPS:
        _check(node.left)
        _check(node.right)
        return
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd, ast.Not)):
        _check(node.operand)
        return
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FUNCS
            and node.args and not node.keywords):
        for arg in node.args:
            _check(arg)
        return
    if isinstance(node, ast.IfExp):
        for part in (node.test, node.body, node.orelse):
            _check(part)
        return
    if isinstance(node, ast.Compare) and all(type(op) in CMPOPS for op in node.ops):
        for part in (node.left, *node.comparators):
            _check(part)
        return
    if isinstance(node, ast.BoolOp):
        for value in node.values:
            _check(value)
        return
    raise BadFormula(f"unsupported syntax ({type(node).__name__}); {SYNTAX_HELP}")


def apply(node, a, b, n, i, d):
    """Evaluate a checked formula. Raises ArithmeticError on /0, huge values, etc."""
    value = _eval(node, {"a": a, "b": b, "n": n, "i": i, "d": d})
    return int(value)


def _eval(node, env):
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        return env[node.id]
    if isinstance(node, ast.BinOp):
        x, y = _eval(node.left, env), _eval(node.right, env)
        if isinstance(node.op, ast.Pow) and not 0 <= y <= MAX_EXPONENT:
            raise ArithmeticError("exponent out of range")
        if abs(x) > MAX_ABS or abs(y) > MAX_ABS:
            raise ArithmeticError("value out of range")
        return _bound(BINOPS[type(node.op)](x, y))
    if isinstance(node, ast.UnaryOp):
        x = _eval(node.operand, env)
        if isinstance(node.op, ast.Not):
            return not x
        return -x if isinstance(node.op, ast.USub) else +x
    if isinstance(node, ast.Call):
        return _bound(FUNCS[node.func.id](*(_eval(arg, env) for arg in node.args)))
    if isinstance(node, ast.IfExp):
        return _eval(node.body, env) if _eval(node.test, env) else _eval(node.orelse, env)
    if isinstance(node, ast.Compare):
        left = _eval(node.left, env)
        for op, comparator in zip(node.ops, node.comparators):
            right = _eval(comparator, env)
            if not CMPOPS[type(op)](left, right):
                return False
            left = right
        return True
    if isinstance(node, ast.BoolOp):
        values = (_eval(v, env) for v in node.values)
        return all(values) if isinstance(node.op, ast.And) else any(values)
    raise ArithmeticError("unexpected node")


def _bound(value):
    if abs(value) > MAX_ABS:
        raise ArithmeticError("value out of range")
    return value
