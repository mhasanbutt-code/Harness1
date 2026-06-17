"""The JEPA ↔ LLM bridge.

A JEPA embedding is a dense vector in representation space — useful to a model,
opaque to a human and not directly consumable by a text-only LLM. This bridge
provides the two interfaces the rest of the system needs:

* ``describe(state)`` -> a compact, human/LLM-readable summary of an embedding
  (magnitude, dominant features, and similarity to named reference states).
  This is what gets injected into the LLM's context window.
* ``project(state)`` -> the raw embedding optionally linearly projected to the
  LLM's hidden size, for setups that splice JEPA vectors directly into the
  model (soft-prompt / adapter style).

Keeping both makes the bridge work whether you feed the LLM *text about* the
state or the *vector itself*.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from harness.config import HarnessConfig
from harness.jepa.encoder import JEPAEncoder


@dataclass
class StateDescription:
    """A textual, LLM-ready view of a JEPA embedding."""

    summary: str
    norm: float
    top_features: list[tuple[int, float]]
    nearest_reference: str | None = None
    nearest_score: float | None = None
    embedding: np.ndarray | None = field(default=None, repr=False)

    def as_prompt_block(self) -> str:
        """Render as a block suitable for injection into an LLM prompt."""
        lines = ["[PERCEIVED STATE]", self.summary, f"- magnitude: {self.norm:.3f}"]
        feats = ", ".join(f"#{i}:{v:+.2f}" for i, v in self.top_features)
        lines.append(f"- dominant features: {feats}")
        if self.nearest_reference is not None:
            lines.append(
                f"- closest known state: '{self.nearest_reference}' "
                f"(similarity {self.nearest_score:.2f})"
            )
        return "\n".join(lines)


class JEPABridge:
    """Turn raw inputs into LLM-consumable state, via a JEPA encoder."""

    def __init__(
        self,
        config: HarnessConfig | None = None,
        encoder: JEPAEncoder | None = None,
        llm_hidden_size: int | None = None,
    ) -> None:
        self.config = config or HarnessConfig()
        self.encoder = encoder or JEPAEncoder(self.config)
        # Named reference embeddings for grounding descriptions ("idle",
        # "error", "running", ...). Populated via `register_reference`.
        self._references: dict[str, np.ndarray] = {}
        # Lazy linear projection JEPA-dim -> LLM hidden size.
        self._llm_hidden_size = llm_hidden_size
        self._projection: np.ndarray | None = None

    # ------------------------------------------------------------------ #
    # Reference states (give the descriptions something to anchor against)
    # ------------------------------------------------------------------ #
    def register_reference(self, name: str, example: Any) -> None:
        """Associate a human-readable name with an example input's embedding."""
        self._references[name] = self.encoder.encode(example)[0]

    # ------------------------------------------------------------------ #
    # Text interface: embedding -> words
    # ------------------------------------------------------------------ #
    def describe(self, state: Any, top_k: int = 5) -> StateDescription:
        """Produce an LLM-readable description of a perceived state."""
        emb = state if isinstance(state, np.ndarray) and state.ndim == 1 else self.encoder.encode(state)[0]
        norm = float(np.linalg.norm(emb))

        idx = np.argsort(-np.abs(emb))[:top_k]
        top_features = [(int(i), float(emb[i])) for i in idx]

        nearest_name, nearest_score = self._nearest_reference(emb)

        summary = self._summarize(norm, top_features, nearest_name, nearest_score)
        return StateDescription(
            summary=summary,
            norm=norm,
            top_features=top_features,
            nearest_reference=nearest_name,
            nearest_score=nearest_score,
            embedding=emb,
        )

    def _nearest_reference(self, emb: np.ndarray) -> tuple[str | None, float | None]:
        if not self._references:
            return None, None
        best_name, best_score = None, -1.0
        en = np.linalg.norm(emb) or 1.0
        for name, ref in self._references.items():
            score = float(np.dot(emb, ref) / (en * (np.linalg.norm(ref) or 1.0)))
            if score > best_score:
                best_name, best_score = name, score
        return best_name, best_score

    @staticmethod
    def _summarize(
        norm: float,
        top_features: list[tuple[int, float]],
        nearest_name: str | None,
        nearest_score: float | None,
    ) -> str:
        if nearest_name is not None and (nearest_score or 0) > 0.6:
            return f"State resembles '{nearest_name}'."
        intensity = "strong" if norm > 1.5 else "moderate" if norm > 0.5 else "weak"
        return f"Novel state with {intensity} activation across {len(top_features)} key features."

    # ------------------------------------------------------------------ #
    # Vector interface: embedding -> LLM hidden space
    # ------------------------------------------------------------------ #
    def project(self, state: Any) -> np.ndarray:
        """Project a state embedding into the LLM hidden size (if configured)."""
        emb = state if isinstance(state, np.ndarray) and state.ndim == 1 else self.encoder.encode(state)[0]
        if self._llm_hidden_size is None or self._llm_hidden_size == emb.shape[0]:
            return emb
        if self._projection is None:
            # Deterministic random projection (Johnson–Lindenstrauss style):
            # preserves geometry well enough for a soft-prompt without training.
            rng = np.random.default_rng(0)
            self._projection = rng.standard_normal(
                (emb.shape[0], self._llm_hidden_size)
            ).astype(np.float32) / np.sqrt(self._llm_hidden_size)
        return emb @ self._projection
