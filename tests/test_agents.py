from harness.agents.agent import Agent
from harness.agents.memory import Memory, Step
from harness.agents.tools import ToolRegistry, default_tools
from harness.config import HarnessConfig
from harness.jepa.interface import JEPABridge


def test_tools_calculator():
    bridge = JEPABridge(HarnessConfig())
    registry = default_tools(bridge, Memory())
    assert registry.call("calculator", "2*(3+4)") == "14"


def test_tools_unknown():
    registry = ToolRegistry([])
    assert "unknown tool" in registry.call("nope", "x")


def test_perceive_tool():
    bridge = JEPABridge(HarnessConfig(jepa_embed_dim=32))
    registry = default_tools(bridge, Memory())
    out = registry.call("perceive", "current")
    assert "[PERCEIVED STATE]" in out


def test_memory_render():
    mem = Memory(goal="g")
    mem.add(Step(thought="t", action="perceive", action_input="current", observation="ok"))
    rendered = mem.render()
    assert "Action: perceive" in rendered and "Observation: ok" in rendered


def test_agent_runs_to_answer():
    agent = Agent(HarnessConfig(jepa_embed_dim=32))
    result = agent.run("Inspect the current state and report health.")
    assert isinstance(result.answer, str) and result.answer
    assert result.stopped_reason in {"final_answer", "max_steps"}
