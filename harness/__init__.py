"""Harness: a local-first JEPA + LLM + agent stack.

Public surface is intentionally small. Import the pieces you need:

    from harness.config import HarnessConfig
    from harness.jepa import JEPAEncoder, JEPABridge
    from harness.llm import LocalLLM
    from harness.agents import Agent
"""

from harness.config import HarnessConfig

__all__ = ["HarnessConfig", "__version__"]
__version__ = "0.1.0"
