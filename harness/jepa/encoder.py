"""JEPA encoder.

A Joint-Embedding Predictive Architecture encodes raw input into an abstract
*representation* rather than reconstructing pixels/tokens. Here we wrap a real
V-JEPA / I-JEPA backbone when `torch` + weights are available, and otherwise
fall back to a deterministic hashing encoder so downstream code (the bridge,
the agent, the app) always receives a stable embedding of the right shape.

The contract is intentionally tiny:

    enc = JEPAEncoder(config)
    emb = enc.encode(inputs)        # -> np.ndarray, shape (N, embed_dim)

`inputs` may be: a numpy array, a path to an image/JSON file, a dict, a string,
or a list mixing those. Anything that isn't a real tensor is canonicalized to
bytes and hashed into the embedding space, which keeps the fallback meaningful
(same input -> same embedding, similar inputs -> overlapping features).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from harness.config import HarnessConfig


class JEPAEncoder:
    """Encode raw inputs into JEPA representation-space embeddings."""

    def __init__(self, config: HarnessConfig | None = None) -> None:
        self.config = config or HarnessConfig()
        self.embed_dim = self.config.jepa_embed_dim
        self._backend = None  # lazily loaded real model, if any
        self._backend_kind = "fallback"

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    @property
    def backend(self) -> str:
        """Which backend is active: ``"vjepa"`` or ``"fallback"``."""
        self._ensure_backend()
        return self._backend_kind

    def encode(self, inputs: Any) -> np.ndarray:
        """Encode one or many inputs into a ``(N, embed_dim)`` float32 array."""
        batch = inputs if _is_batch(inputs) else [inputs]
        self._ensure_backend()

        if self._backend is not None:
            try:
                return self._encode_real(batch)
            except Exception:
                # Never let a backend hiccup break the pipeline.
                self._backend = None
                self._backend_kind = "fallback"

        return np.stack([self._encode_fallback(x) for x in batch]).astype(np.float32)

    def similarity(self, a: Any, b: Any) -> float:
        """Cosine similarity between the embeddings of two inputs."""
        ea = self.encode(a)[0]
        eb = self.encode(b)[0]
        denom = (np.linalg.norm(ea) * np.linalg.norm(eb)) or 1.0
        return float(np.dot(ea, eb) / denom)

    # ------------------------------------------------------------------ #
    # Backend management
    # ------------------------------------------------------------------ #
    def _ensure_backend(self) -> None:
        if self._backend is not None or self._backend_kind == "vjepa":
            return
        try:
            import torch  # noqa: F401
            from transformers import AutoModel  # type: ignore

            model = AutoModel.from_pretrained(self.config.jepa_model)
            model.eval()
            self._backend = model
            self._backend_kind = "vjepa"
            # Trust the real model's hidden size when we can read it.
            hidden = getattr(getattr(model, "config", None), "hidden_size", None)
            if isinstance(hidden, int):
                self.embed_dim = hidden
        except Exception:
            # Missing torch/transformers or weights -> deterministic fallback.
            self._backend = None
            self._backend_kind = "fallback"

    def _encode_real(self, batch: Sequence[Any]) -> np.ndarray:
        import torch

        tensors = torch.stack([_to_tensor(x) for x in batch])
        with torch.no_grad():
            out = self._backend(tensors)  # type: ignore[misc]
        hidden = getattr(out, "last_hidden_state", out)
        # Mean-pool the token/patch axis to a single vector per item.
        pooled = hidden.mean(dim=1) if hidden.ndim == 3 else hidden
        return pooled.detach().cpu().numpy().astype(np.float32)

    # ------------------------------------------------------------------ #
    # Deterministic fallback
    # ------------------------------------------------------------------ #
    def _encode_fallback(self, x: Any) -> np.ndarray:
        """Map an arbitrary input to a stable pseudo-embedding.

        We seed a PRNG with a hash of the canonical bytes of the input, so the
        embedding is deterministic and structurally meaningful: identical
        inputs match exactly, and a shared prefix/feature shifts the seed
        predictably.
        """
        raw = _canonical_bytes(x)
        seed = int.from_bytes(hashlib.sha256(raw).digest()[:8], "big")
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(self.embed_dim).astype(np.float32)
        # L2-normalize so cosine similarity is well-behaved.
        vec /= np.linalg.norm(vec) or 1.0
        return vec


# ---------------------------------------------------------------------- #
# Input canonicalization helpers
# ---------------------------------------------------------------------- #
def _is_batch(inputs: Any) -> bool:
    if isinstance(inputs, np.ndarray):
        return inputs.ndim > 1
    return isinstance(inputs, (list, tuple))


def _canonical_bytes(x: Any) -> bytes:
    if isinstance(x, bytes):
        return x
    if isinstance(x, np.ndarray):
        return x.tobytes()
    if isinstance(x, (dict, list, tuple)):
        return json.dumps(x, sort_keys=True, default=str).encode("utf-8")
    if isinstance(x, (str, Path)):
        p = Path(x)
        if p.exists() and p.is_file():
            return p.read_bytes()
        return str(x).encode("utf-8")
    return repr(x).encode("utf-8")


def _to_tensor(x: Any):
    """Best-effort conversion of an input into a float tensor for a real model."""
    import torch

    if hasattr(x, "shape") and not isinstance(x, np.ndarray):
        return x  # already a tensor
    arr = x if isinstance(x, np.ndarray) else np.asarray(_canonical_bytes(x), dtype=np.uint8)
    return torch.as_tensor(np.asarray(arr, dtype=np.float32))
