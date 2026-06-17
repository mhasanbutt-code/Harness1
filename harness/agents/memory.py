"""Working memory for the agent.

A deliberately small structure: an ordered transcript of steps plus a scratch
key/value store the agent (and tools) can read and write. Enough to support a
multi-step loop and to render recent context back into the prompt without
unbounded growth.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Step:
    thought: str = ""
    action: str = ""
    action_input: str = ""
    observation: str = ""


@dataclass
class Memory:
    goal: str = ""
    steps: list[Step] = field(default_factory=list)
    scratch: dict[str, Any] = field(default_factory=dict)
    max_render_steps: int = 6

    def add(self, step: Step) -> None:
        self.steps.append(step)

    def remember(self, key: str, value: Any) -> None:
        self.scratch[key] = value

    def recall(self, key: str, default: Any = None) -> Any:
        return self.scratch.get(key, default)

    def render(self) -> str:
        """Render recent steps as a ReAct transcript for the prompt."""
        recent = self.steps[-self.max_render_steps:]
        lines: list[str] = []
        for s in recent:
            if s.thought:
                lines.append(f"Thought: {s.thought}")
            if s.action:
                lines.append(f"Action: {s.action}")
                lines.append(f"Action Input: {s.action_input}")
            if s.observation:
                lines.append(f"Observation: {s.observation}")
        return "\n".join(lines)
