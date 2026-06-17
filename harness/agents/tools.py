"""Tools the agent can call.

A tool is a named, documented callable with a single-string input and a
single-string output (kept stringly-typed so any LLM can drive it through the
ReAct text protocol). The registry renders tool docs into the prompt and
dispatches actions.

Built-in tools:
* ``perceive``    — run the JEPA bridge on an input and describe the state.
* ``calculator``  — safe arithmetic.
* ``remember`` / ``recall`` — read/write the agent's scratch memory.
* ``final``       — sentinel handled by the loop (not registered as callable).
"""

from __future__ import annotations

import ast
import operator
from dataclasses import dataclass
from typing import Callable

from harness.agents.memory import Memory
from harness.jepa.interface import JEPABridge


@dataclass
class Tool:
    name: str
    description: str
    func: Callable[[str], str]

    def __call__(self, arg: str) -> str:
        return self.func(arg)


class ToolRegistry:
    """Holds tools and renders/dispatches them for the agent loop."""

    def __init__(self, tools: list[Tool] | None = None) -> None:
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def has(self, name: str) -> bool:
        return name in self._tools

    def names(self) -> list[str]:
        return list(self._tools)

    def render(self) -> str:
        """Render the tool list for injection into the prompt."""
        return "\n".join(f"- {t.name}: {t.description}" for t in self._tools.values())

    def call(self, name: str, arg: str) -> str:
        if name not in self._tools:
            return f"Error: unknown tool '{name}'. Available: {', '.join(self.names())}"
        try:
            return self._tools[name](arg)
        except Exception as e:  # tools must never crash the loop
            return f"Error running '{name}': {e}"


# ---------------------------------------------------------------------- #
# Built-in tool implementations
# ---------------------------------------------------------------------- #
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _safe_eval(expr: str) -> str:
    """Evaluate an arithmetic expression without using `eval`."""

    def _ev(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
            return _BIN_OPS[type(node.op)](_ev(node.left), _ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
            return _UNARY_OPS[type(node.op)](_ev(node.operand))
        raise ValueError("unsupported expression")

    tree = ast.parse(expr, mode="eval")
    return str(_ev(tree))


def make_perceive_tool(bridge: JEPABridge) -> Tool:
    """A tool that perceives an input via JEPA and describes the state."""

    def _perceive(arg: str) -> str:
        desc = bridge.describe(arg.strip() or "current")
        return desc.as_prompt_block()

    return Tool(
        name="perceive",
        description="Perceive an input via the JEPA encoder; returns a state description. "
        "Input: a path, a short state string, or 'current'.",
        func=_perceive,
    )


def make_memory_tools(memory: Memory) -> list[Tool]:
    def _remember(arg: str) -> str:
        if "=" not in arg:
            return "Error: use 'key=value'."
        key, value = arg.split("=", 1)
        memory.remember(key.strip(), value.strip())
        return f"Stored {key.strip()!r}."

    def _recall(arg: str) -> str:
        val = memory.recall(arg.strip())
        return f"{arg.strip()} = {val}" if val is not None else f"No memory for {arg.strip()!r}."

    return [
        Tool("remember", "Store a fact in scratch memory. Input: 'key=value'.", _remember),
        Tool("recall", "Recall a stored fact. Input: the key.", _recall),
    ]


def default_tools(bridge: JEPABridge, memory: Memory) -> ToolRegistry:
    """Assemble the standard tool set wired to a bridge and memory."""
    registry = ToolRegistry(
        [
            make_perceive_tool(bridge),
            Tool("calculator", "Evaluate arithmetic. Input: an expression like '2*(3+4)'.", _safe_eval),
        ]
    )
    for t in make_memory_tools(memory):
        registry.register(t)
    return registry
