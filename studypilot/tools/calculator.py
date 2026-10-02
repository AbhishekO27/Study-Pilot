"""A calculator tool the agent can call for arithmetic.

LLMs are unreliable at arithmetic. Giving the model a calculator and letting
it decide when to use one is the smallest real example of "agent uses a tool".
"""

from __future__ import annotations

import ast
import operator

from langchain_core.tools import tool

# Only these operators are allowed. Anything else in the expression is rejected.
# We walk the parsed syntax tree instead of calling eval(), because eval()
# would happily run "__import__('os').system('...')" if the model sent it.
_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
}
_UNARY_OPS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

# Guard against "9 ** 9 ** 9", which would never finish.
_MAX_POWER = 1000


def _evaluate(node: ast.AST) -> float:
    """Recursively evaluate one node of the parsed expression."""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value

    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_evaluate(node.operand))

    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left = _evaluate(node.left)
        right = _evaluate(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_POWER:
            raise ValueError(f"exponent too large (max {_MAX_POWER})")
        return _BINARY_OPS[type(node.op)](left, right)

    # Names, function calls, attribute access, etc. all end up here.
    raise ValueError(f"unsupported syntax: {type(node).__name__}")


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression such as "17 * 23" or "(1 + 0.9 / 8) ** -1".

    Supports + - * / ** % and parentheses. Use this for any arithmetic
    instead of computing in your head.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _evaluate(tree.body)
    except ZeroDivisionError:
        return "Error: division by zero"
    except (ValueError, SyntaxError) as exc:
        # Return the error as text. The model reads it and can try again.
        # Raising here would crash the agent loop instead.
        return f"Error: could not evaluate '{expression}': {exc}"

    # Show whole numbers without a trailing .0 so "17 * 23" reads as 391.
    if isinstance(result, float) and result.is_integer():
        return str(int(result))
    return str(result)
