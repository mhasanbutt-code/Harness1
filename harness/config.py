"""Central configuration for the whole harness.

A single dataclass drives every component so that the JEPA encoder, the LLM,
the trainer, and the agent all agree on dimensions, paths, and model ids.
Load from a JSON file and/or environment variables; everything has a sane
default so `HarnessConfig()` alone is enough to run the fallback pipeline.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LoRAConfig:
    """LoRA / QLoRA hyperparameters for parameter-efficient fine-tuning."""

    r: int = 16
    alpha: int = 32
    dropout: float = 0.05
    # Attention/MLP projection names common to Llama/Qwen/Mistral families.
    target_modules: list[str] = field(
        default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"]
    )
    # 4-bit quantized base weights (QLoRA). Ignored if bitsandbytes is absent.
    load_in_4bit: bool = True


@dataclass
class HarnessConfig:
    """Configuration shared across every component."""

    # --- Local LLM ---
    base_model: str = "Qwen/Qwen2.5-0.5B-Instruct"
    device: str = "auto"  # "auto" | "cpu" | "cuda" | "mps"
    max_new_tokens: int = 256
    temperature: float = 0.7

    # --- JEPA ---
    # A real V-JEPA / I-JEPA id can go here; if it can't be loaded the encoder
    # falls back to a deterministic stub of the same `jepa_embed_dim`.
    jepa_model: str = "facebook/vjepa2-vitl-fpc64-256"
    jepa_embed_dim: int = 1024

    # --- Training ---
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    learning_rate: float = 2e-4
    num_epochs: int = 3
    batch_size: int = 2
    grad_accum: int = 8
    max_seq_len: int = 1024

    # --- Agent loop ---
    max_agent_steps: int = 8

    # --- Paths ---
    data_dir: Path = Path("data")
    output_dir: Path = Path("outputs")
    adapter_dir: Path = Path("outputs/adapter")

    # ------------------------------------------------------------------ #
    # Construction helpers
    # ------------------------------------------------------------------ #
    @classmethod
    def load(cls, path: str | os.PathLike[str] | None = None) -> "HarnessConfig":
        """Build a config from an optional JSON file, then env overrides.

        Environment variables are upper-cased field names prefixed with
        ``HARNESS_`` (e.g. ``HARNESS_BASE_MODEL``, ``HARNESS_DEVICE``).
        """
        data: dict[str, Any] = {}
        if path is not None and Path(path).exists():
            data = json.loads(Path(path).read_text())

        lora_data = data.pop("lora", None)
        cfg = cls(**data)
        if lora_data:
            cfg.lora = LoRAConfig(**lora_data)

        cfg._apply_env()
        cfg._coerce_paths()
        return cfg

    def _apply_env(self) -> None:
        for f in self.__dataclass_fields__:  # type: ignore[attr-defined]
            if f == "lora":
                continue
            env_key = f"HARNESS_{f.upper()}"
            if env_key in os.environ:
                raw = os.environ[env_key]
                current = getattr(self, f)
                setattr(self, f, _coerce(raw, type(current)))

    def _coerce_paths(self) -> None:
        self.data_dir = Path(self.data_dir)
        self.output_dir = Path(self.output_dir)
        self.adapter_dir = Path(self.adapter_dir)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("data_dir", "output_dir", "adapter_dir"):
            d[k] = str(d[k])
        return d

    def save(self, path: str | os.PathLike[str]) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    def resolve_device(self) -> str:
        """Resolve ``"auto"`` to the best available backend."""
        if self.device != "auto":
            return self.device
        try:  # torch is optional
            import torch

            if torch.cuda.is_available():
                return "cuda"
            if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"


def _coerce(raw: str, target_type: type) -> Any:
    """Coerce an environment string into the target field type."""
    if target_type is bool:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if target_type is int:
        return int(raw)
    if target_type is float:
        return float(raw)
    if target_type is Path:
        return Path(raw)
    return raw
