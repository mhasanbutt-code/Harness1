"""End-to-end walkthrough of the harness, using fallbacks (no heavy deps).

Run:  python examples/run_pipeline.py
"""

from harness.agents.agent import Agent
from harness.config import HarnessConfig
from harness.jepa.interface import JEPABridge
from harness.llm.eval import evaluate_file
from harness.llm.model import LocalLLM


def main() -> None:
    config = HarnessConfig()

    # 1. JEPA: perceive a state and describe it for the LLM.
    bridge = JEPABridge(config)
    bridge.register_reference("running", {"status": "running"})
    bridge.register_reference("error", {"status": "error"})
    desc = bridge.describe("examples/state.json")
    print("=== JEPA perception ===")
    print(desc.as_prompt_block())
    print(f"(encoder backend: {bridge.encoder.backend})\n")

    # 2. LLM: a direct generation.
    llm = LocalLLM(config).load()
    print("=== LLM ===")
    print(f"backend: {llm.kind}")
    print(llm.generate("### Instruction:\nName the four harness pieces.\n### Response:\n"), "\n")

    # 3. Agent: run the ReAct loop (will use the 'perceive' tool).
    print("=== Agent ===")
    agent = Agent(config, llm=llm, bridge=bridge)
    result = agent.run("Inspect the current harness state and report whether it looks healthy.")
    print("answer:", result.answer)
    print("steps:", result.num_steps, "stopped:", result.stopped_reason, "\n")

    # 4. Eval: score the LLM on the sample data.
    print("=== Eval ===")
    print(evaluate_file("data/sample_train.jsonl", llm=llm, config=config).summary())


if __name__ == "__main__":
    main()
