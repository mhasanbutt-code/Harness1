"""Local causal-LM wrapper.

Loads an open-weights instruct model with `transformers`, optionally applying a
trained LoRA adapter. When `transformers`/`torch` or the weights are absent, it
falls back to a deterministic, dependency-free `_EchoModel` so the agent loop
and the app remain runnable. The fallback is not smart, but it is *consistent*,
which is what the rest of the system needs to be testable.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from harness.config import HarnessConfig


class LocalLLM:
    """A thin generation interface over a local model (or an echo fallback)."""

    def __init__(self, config: HarnessConfig | None = None) -> None:
        self.config = config or HarnessConfig()
        self._model = None
        self._tokenizer = None
        self._kind = "uninitialized"

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    @property
    def kind(self) -> str:
        """``"transformers"`` once real weights load, else ``"echo"``."""
        if self._kind == "uninitialized":
            self.load()
        return self._kind

    def load(self, adapter_dir: str | Path | None = None) -> "LocalLLM":
        """Load the base model (and optional LoRA adapter), or the fallback."""
        adapter = Path(adapter_dir) if adapter_dir else self.config.adapter_dir
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            tok = AutoTokenizer.from_pretrained(self.config.base_model)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            model = AutoModelForCausalLM.from_pretrained(
                self.config.base_model,
                torch_dtype="auto",
                device_map=self.config.resolve_device(),
            )
            if adapter.exists():
                try:
                    from peft import PeftModel

                    model = PeftModel.from_pretrained(model, str(adapter))
                except Exception:
                    pass  # adapter optional; continue with base weights
            model.eval()
            self._model, self._tokenizer, self._kind = model, tok, "transformers"
            self._torch = torch
        except Exception:
            self._model, self._tokenizer, self._kind = _EchoModel(), None, "echo"
        return self

    # ------------------------------------------------------------------ #
    # Generation
    # ------------------------------------------------------------------ #
    def generate(self, prompt: str, **overrides: Any) -> str:
        """Generate a completion for a raw prompt string."""
        if self._kind == "uninitialized":
            self.load()

        if self._kind == "echo":
            return self._model.generate(prompt)  # type: ignore[union-attr]

        max_new = overrides.get("max_new_tokens", self.config.max_new_tokens)
        temperature = overrides.get("temperature", self.config.temperature)
        tok, model, torch = self._tokenizer, self._model, self._torch

        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new,
                do_sample=temperature > 0,
                temperature=max(temperature, 1e-5),
                pad_token_id=tok.pad_token_id,
            )
        text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return text.strip()

    def chat(self, messages: list[dict[str, str]], **overrides: Any) -> str:
        """Generate from a list of ``{"role", "content"}`` messages."""
        if self.kind == "transformers" and hasattr(self._tokenizer, "apply_chat_template"):
            prompt = self._tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            prompt = _format_chat(messages)
        return self.generate(prompt, **overrides)


def _format_chat(messages: list[dict[str, str]]) -> str:
    """Minimal chat formatting used by the fallback and as a template default."""
    parts = [f"{m['role'].upper()}: {m['content']}" for m in messages]
    parts.append("ASSISTANT:")
    return "\n".join(parts)


class _EchoModel:
    """Deterministic stand-in model.

    It produces a plausible, *parseable* response so the agent's ReAct parser
    has something to work with even without real weights. It looks for a goal
    in the prompt and emits a final answer, which keeps the loop terminating.
    """

    def generate(self, prompt: str) -> str:
        lowered = prompt.lower()
        # If the agent prompt asks for a tool, occasionally suggest 'perceive'
        # so the JEPA path is exercised; otherwise answer directly.
        if "available tools" in lowered and "perceive" in lowered and "perceived state" not in lowered:
            return (
                "Thought: I should inspect the current state first.\n"
                "Action: perceive\n"
                "Action Input: current"
            )
        goal = _extract_goal(prompt)
        return (
            "Thought: I have enough information to respond.\n"
            f"Final Answer: [echo-model] Acknowledged: {goal}"
        )


def _extract_goal(prompt: str) -> str:
    for marker in ("GOAL:", "USER:", "Goal:"):
        if marker in prompt:
            tail = prompt.split(marker, 1)[1].strip()
            return tail.splitlines()[0][:200] if tail else "(empty)"
    return prompt.strip().splitlines()[-1][:200] if prompt.strip() else "(empty)"
