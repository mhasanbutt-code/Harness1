"""The agent orchestration loop.

A compact ReAct controller: the LLM emits `Thought / Action / Action Input`,
the loop executes the named tool, appends the `Observation`, and repeats until
the model emits a `Final Answer` or the step budget is hit. JEPA perception is
just another tool (`perceive`), so the agent can choose to look at the world
before acting.

This ties the first three pieces together: the agent reasons with the **LLM**,
perceives with the **JEPA bridge**, and acts through **tools**.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from harness.agents.memory import Memory, Step
from harness.agents.tools import ToolRegistry, default_tools
from harness.config import HarnessConfig
from harness.jepa.interface import JEPABridge
from harness.llm.model import LocalLLM

SYSTEM_PROMPT = """You are the harness agent. Solve the user's goal using the \
available tools. Reason step by step.

Respond in this exact format, one block at a time:
Thought: <your reasoning>
Action: <one tool name from the list, or 'final'>
Action Input: <input for the tool, or your answer if Action is 'final'>

When you are done, use:
Thought: <reasoning>
Final Answer: <the answer>

Available tools:
{tools}
"""

_ACTION_RE = re.compile(r"Action:\s*(.+)", re.IGNORECASE)
_INPUT_RE = re.compile(r"Action Input:\s*(.+)", re.IGNORECASE | re.DOTALL)
_THOUGHT_RE = re.compile(r"Thought:\s*(.+)", re.IGNORECASE)
_FINAL_RE = re.compile(r"Final Answer:\s*(.+)", re.IGNORECASE | re.DOTALL)


@dataclass
class AgentResult:
    answer: str
    steps: list[Step] = field(default_factory=list)
    stopped_reason: str = "final_answer"  # or "max_steps"

    @property
    def num_steps(self) -> int:
        return len(self.steps)


class Agent:
    """A tool-using, JEPA-perceiving agent over a local LLM."""

    def __init__(
        self,
        config: HarnessConfig | None = None,
        llm: LocalLLM | None = None,
        bridge: JEPABridge | None = None,
        tools: ToolRegistry | None = None,
        memory: Memory | None = None,
    ) -> None:
        self.config = config or HarnessConfig()
        self.llm = llm or LocalLLM(self.config)
        self.bridge = bridge or JEPABridge(self.config)
        self.memory = memory or Memory()
        self.tools = tools or default_tools(self.bridge, self.memory)

    # ------------------------------------------------------------------ #
    def run(self, goal: str) -> AgentResult:
        """Drive the ReAct loop until a final answer or the step budget."""
        self.memory.goal = goal
        system = SYSTEM_PROMPT.format(tools=self.tools.render())

        for _ in range(self.config.max_agent_steps):
            prompt = self._build_prompt(system, goal)
            raw = self.llm.generate(prompt)

            final = _FINAL_RE.search(raw)
            if final:
                answer = final.group(1).strip()
                self.memory.add(Step(thought=_first(_THOUGHT_RE, raw), observation=""))
                return AgentResult(answer=answer, steps=self.memory.steps, stopped_reason="final_answer")

            action = _first(_ACTION_RE, raw)
            action_input = _first(_INPUT_RE, raw)
            thought = _first(_THOUGHT_RE, raw)

            if not action or action.lower() == "final":
                # Model produced no actionable step; treat its text as answer.
                return AgentResult(
                    answer=action_input or raw.strip(),
                    steps=self.memory.steps,
                    stopped_reason="final_answer",
                )

            observation = self.tools.call(action.strip(), action_input.strip())
            self.memory.add(
                Step(thought=thought, action=action.strip(), action_input=action_input.strip(), observation=observation)
            )

        return AgentResult(
            answer=self._fallback_answer(),
            steps=self.memory.steps,
            stopped_reason="max_steps",
        )

    # ------------------------------------------------------------------ #
    def _build_prompt(self, system: str, goal: str) -> str:
        transcript = self.memory.render()
        parts = [system, "", f"GOAL: {goal}", ""]
        if transcript:
            parts += [transcript, ""]
        parts.append("Continue with the next Thought/Action or a Final Answer.")
        return "\n".join(parts)

    def _fallback_answer(self) -> str:
        last = next((s.observation for s in reversed(self.memory.steps) if s.observation), "")
        return last or "Reached step budget without a final answer."


def _first(pattern: re.Pattern[str], text: str) -> str:
    m = pattern.search(text)
    if not m:
        return ""
    # Stop a greedy DOTALL match at the next field marker.
    value = m.group(1).strip()
    for marker in ("\nThought:", "\nAction:", "\nObservation:", "\nFinal Answer:"):
        if marker.strip() in value:
            value = value.split(marker.strip())[0].strip()
    return value.splitlines()[0].strip() if pattern in (_ACTION_RE,) else value
