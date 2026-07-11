"""Safe boolean/arithmetic expression evaluation over a fact table.

This is the benchmark's entire "policy language": every gate, classification
rule, action rule, and decision-slot rule is a `when:` expression evaluated
against a flat dict of facts. Expressions are parsed with `ast` and only a
whitelisted node set is allowed — no calls, no attributes, no subscripts, no
imports. Evaluation is total and deterministic: referencing a fact that is not
in the table raises, it never guesses.

The same expression string is rendered verbatim into the scenario prompt as
the stated policy, which is what makes gold derivable and non-divergent from
what the model was told.
"""
from __future__ import annotations

import ast
from typing import Any

_ALLOWED_NODES = (
    ast.Expression,
    ast.BoolOp, ast.And, ast.Or,
    ast.UnaryOp, ast.Not, ast.USub,
    ast.BinOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod,
    ast.Compare, ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE,
    ast.In, ast.NotIn,
    ast.Name, ast.Load, ast.Constant,
    ast.List, ast.Tuple,
)


class ExprError(ValueError):
    pass


def _check(node: ast.AST, expr: str) -> None:
    for child in ast.walk(node):
        if not isinstance(child, _ALLOWED_NODES):
            raise ExprError(
                f"disallowed syntax {type(child).__name__!r} in policy expression: {expr!r}"
            )


def evaluate(expr: str, facts: dict[str, Any]) -> Any:
    """Evaluate `expr` against `facts`. Unknown names raise ExprError."""
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise ExprError(f"invalid policy expression {expr!r}: {exc}") from exc
    _check(tree, expr)

    def _eval(node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in facts:
                raise ExprError(f"unknown fact {node.id!r} in expression {expr!r}")
            return facts[node.id]
        if isinstance(node, (ast.List, ast.Tuple)):
            return [_eval(e) for e in node.elts]
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return all(_eval(v) for v in node.values)
            return any(_eval(v) for v in node.values)
        if isinstance(node, ast.UnaryOp):
            val = _eval(node.operand)
            return (not val) if isinstance(node.op, ast.Not) else -val
        if isinstance(node, ast.BinOp):
            left, right = _eval(node.left), _eval(node.right)
            ops = {
                ast.Add: lambda a, b: a + b,
                ast.Sub: lambda a, b: a - b,
                ast.Mult: lambda a, b: a * b,
                ast.Div: lambda a, b: a / b,
                ast.FloorDiv: lambda a, b: a // b,
                ast.Mod: lambda a, b: a % b,
            }
            return ops[type(node.op)](left, right)
        if isinstance(node, ast.Compare):
            left = _eval(node.left)
            for op, comparator in zip(node.ops, node.comparators):
                right = _eval(comparator)
                ops = {
                    ast.Eq: lambda a, b: a == b,
                    ast.NotEq: lambda a, b: a != b,
                    ast.Lt: lambda a, b: a < b,
                    ast.LtE: lambda a, b: a <= b,
                    ast.Gt: lambda a, b: a > b,
                    ast.GtE: lambda a, b: a >= b,
                    ast.In: lambda a, b: a in b,
                    ast.NotIn: lambda a, b: a not in b,
                }
                if not ops[type(op)](left, right):
                    return False
                left = right
            return True
        raise ExprError(f"unsupported node {type(node).__name__!r} in {expr!r}")

    return _eval(tree)


def expr_fact_names(expr: str) -> set[str]:
    """All fact names referenced by `expr` (for lint: every name must be a
    stated fact or a declared unknown)."""
    tree = ast.parse(expr, mode="eval")
    _check(tree, expr)
    return {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
