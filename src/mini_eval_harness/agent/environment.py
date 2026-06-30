from __future__ import annotations

import ast
import copy
import operator
from dataclasses import dataclass
from typing import Any, Callable

from mini_eval_harness.rag.retriever import tokenize_text


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    output: dict[str, Any]
    error: str | None = None


ToolHandler = Callable[[dict[str, Any]], ToolResult]


TOOL_DESCRIPTIONS: dict[str, str] = {
    "get_user_profile": (
        "get_user_profile(user_id): return a user profile from state.users."
    ),
    "send_message": (
        "send_message(user_id, text): append a simulated message to state.messages."
    ),
    "update_todo": (
        "update_todo(item_id, status, note?): update a todo status and optional note."
    ),
    "calculator": "calculator(expression): evaluate a basic arithmetic expression.",
    "search_docs": "search_docs(query, top_k?): keyword search over state.docs.",
}


class AgentEnvironment:
    def __init__(self, initial_state: dict[str, Any], allowed_tools: list[str]) -> None:
        self.initial_state = copy.deepcopy(initial_state)
        self.state = copy.deepcopy(initial_state)
        self.allowed_tools = set(allowed_tools)
        self.handlers: dict[str, ToolHandler] = {
            "get_user_profile": self._get_user_profile,
            "send_message": self._send_message,
            "update_todo": self._update_todo,
            "calculator": self._calculator,
            "search_docs": self._search_docs,
        }

    def execute(self, tool_name: str, args: dict[str, Any]) -> ToolResult:
        if tool_name not in self.allowed_tools:
            return ToolResult(
                ok=False,
                output={},
                error=f"Tool is not allowed for this task: {tool_name}",
            )
        handler = self.handlers.get(tool_name)
        if handler is None:
            return ToolResult(
                ok=False,
                output={},
                error=f"Unknown tool: {tool_name}",
            )
        try:
            return handler(args)
        except (KeyError, TypeError, ValueError) as exc:
            return ToolResult(
                ok=False,
                output={},
                error=f"{type(exc).__name__}: {exc}",
            )

    def tool_descriptions(self) -> list[str]:
        return [
            TOOL_DESCRIPTIONS.get(tool_name, tool_name)
            for tool_name in sorted(self.allowed_tools)
        ]

    def snapshot(self) -> dict[str, Any]:
        return copy.deepcopy(self.state)

    def _get_user_profile(self, args: dict[str, Any]) -> ToolResult:
        user_id = require_string(args, "user_id")
        users = self.state.setdefault("users", {})
        if user_id not in users:
            raise ValueError(f"Unknown user_id: {user_id}")
        return ToolResult(ok=True, output={"user_id": user_id, "profile": users[user_id]})

    def _send_message(self, args: dict[str, Any]) -> ToolResult:
        user_id = require_string(args, "user_id")
        text = require_string(args, "text")
        users = self.state.setdefault("users", {})
        if user_id not in users:
            raise ValueError(f"Unknown user_id: {user_id}")

        messages = self.state.setdefault("messages", [])
        if not isinstance(messages, list):
            raise ValueError("state.messages must be a list")
        message = {"to": user_id, "text": text}
        messages.append(message)
        return ToolResult(ok=True, output={"message": message})

    def _update_todo(self, args: dict[str, Any]) -> ToolResult:
        item_id = require_string(args, "item_id")
        status = require_string(args, "status")
        note = args.get("note")
        todos = self.state.setdefault("todos", {})
        if item_id not in todos:
            raise ValueError(f"Unknown item_id: {item_id}")

        todo = todos[item_id]
        if not isinstance(todo, dict):
            raise ValueError(f"Todo must be an object: {item_id}")
        todo["status"] = status
        if note is not None:
            todo["note"] = str(note)
        return ToolResult(ok=True, output={"item_id": item_id, "todo": todo})

    def _calculator(self, args: dict[str, Any]) -> ToolResult:
        expression = require_string(args, "expression")
        result = safe_eval_arithmetic(expression)
        return ToolResult(
            ok=True,
            output={"expression": expression, "result": result},
        )

    def _search_docs(self, args: dict[str, Any]) -> ToolResult:
        query = require_string(args, "query")
        top_k = int(args.get("top_k", 3))
        docs = self.state.setdefault("docs", [])
        if not isinstance(docs, list):
            raise ValueError("state.docs must be a list")

        query_tokens = set(tokenize_text(query))
        scored: list[tuple[int, dict[str, Any]]] = []
        for doc in docs:
            if not isinstance(doc, dict):
                continue
            text = str(doc.get("text", ""))
            score = len(query_tokens.intersection(tokenize_text(text)))
            scored.append((score, doc))

        ranked = sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]
        return ToolResult(
            ok=True,
            output={
                "query": query,
                "results": [
                    {
                        "id": str(doc.get("id", "")),
                        "text": str(doc.get("text", "")),
                        "score": score,
                    }
                    for score, doc in ranked
                    if score > 0
                ],
            },
        )


def require_string(args: dict[str, Any], key: str) -> str:
    value = args.get(key)
    if value is None:
        raise ValueError(f"Missing required arg: {key}")
    return str(value)


def safe_eval_arithmetic(expression: str) -> int | float:
    allowed_binary_ops = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
    }
    allowed_unary_ops = {
        ast.UAdd: operator.pos,
        ast.USub: operator.neg,
    }

    def eval_node(node: ast.AST) -> int | float:
        if isinstance(node, ast.Expression):
            return eval_node(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in allowed_binary_ops:
            left = eval_node(node.left)
            right = eval_node(node.right)
            return allowed_binary_ops[type(node.op)](left, right)
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed_unary_ops:
            return allowed_unary_ops[type(node.op)](eval_node(node.operand))
        raise ValueError("Only basic arithmetic expressions are allowed")

    parsed = ast.parse(expression, mode="eval")
    return eval_node(parsed)
