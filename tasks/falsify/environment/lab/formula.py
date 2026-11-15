"""Safe parser and evaluator for operator formulas such as "a + b**2" or "max(a, b)".

Formulas are checked against an AST whitelist and never passed to eval(). They may use
integers, a, b, the context variables n (operands in the expression), i (operator position)
and d (nesting depth), + - * // % **, unary minus, abs/max/min, comparisons, and/or/not
and "x if cond else y". environment/lab/formula.py and tests/formula.py must be identical.
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
    """Raised for a formula outside the allowed syntax."""


def compile_formula(src):
    """Parse and check a formula.

    Args:
        src: The formula string.

    Returns:
        The checked AST expression node.

    Raises:
        BadFormula: If the formula is empty, too long or uses unsupported syntax.
    """
    if not isinstance(src, str) or not src.strip() or len(src) > MAX_LEN:
        raise BadFormula(f"a formula must be a non-empty string of at most {MAX_LEN} characters")
    try:
        tree = ast.parse(src.strip(), mode="eval")
    except SyntaxError as e:
        raise BadFormula(f"not a valid expression ({e.msg})") from None
    _check(tree.body)
    return tree.body


def _check(node):
    """Raise BadFormula unless every node of the expression is whitelisted."""
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
    """Evaluate a checked formula.

    Args:
        node: A node returned by compile_formula.
        a: The left operand.
        b: The right operand.
        n: The number of operands in the whole expression.
        i: The operator's position, counted from the left starting at 1.
        d: The operation's nesting depth, 0 for the operation evaluated last.

    Returns:
        The integer result.

    Raises:
        ArithmeticError: On division by zero, an out-of-range exponent or a value beyond 10**15.
    """
    return int(_eval(node, {"a": a, "b": b, "n": n, "i": i, "d": d}))


def _eval(node, env):
    """Recursively evaluate a checked node in the given variable environment."""
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
    """Return value, or raise ArithmeticError if it exceeds 10**15 in magnitude."""
    if abs(value) > MAX_ABS:
        raise ArithmeticError("value out of range")
    return value
