"""Evaluation loop for the local LLM.

Runs the model over held-out instruction examples and reports lightweight,
dependency-free metrics: exact-match rate and mean token-overlap (F1). This is
deliberately simple so it runs anywhere — swap in task-specific metrics as your
harness matures. The eval loop closes the training cycle: train -> eval ->
inspect failures -> feed failures back as data.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from harness.config import HarnessConfig
from harness.llm.data import Example, read_jsonl
from harness.llm.model import LocalLLM

_WORD = re.compile(r"\w+")


@dataclass
class EvalResult:
    n: int
    exact_match: float
    token_f1: float
    samples: list[dict] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"n={self.n}  exact_match={self.exact_match:.3f}  "
            f"token_f1={self.token_f1:.3f}"
        )


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def token_f1(pred: str, gold: str) -> float:
    """Bag-of-words F1 between prediction and reference."""
    p, g = _tokens(pred), _tokens(gold)
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    common: dict[str, int] = {}
    gold_counts: dict[str, int] = {}
    for t in g:
        gold_counts[t] = gold_counts.get(t, 0) + 1
    overlap = 0
    for t in p:
        if gold_counts.get(t, 0) - common.get(t, 0) > 0:
            common[t] = common.get(t, 0) + 1
            overlap += 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(p)
    recall = overlap / len(g)
    return 2 * precision * recall / (precision + recall)


def evaluate(
    examples: list[Example],
    llm: LocalLLM | None = None,
    config: HarnessConfig | None = None,
    keep_samples: int = 5,
) -> EvalResult:
    """Evaluate ``llm`` over ``examples`` and return aggregate metrics."""
    config = config or HarnessConfig()
    llm = llm or LocalLLM(config).load()

    exact = 0
    f1_total = 0.0
    samples: list[dict] = []

    for ex in examples:
        pred = llm.generate(ex.prompt(), temperature=0.0).strip()
        gold = ex.response.strip()
        em = int(pred == gold)
        f1 = token_f1(pred, gold)
        exact += em
        f1_total += f1
        if len(samples) < keep_samples:
            samples.append(
                {"instruction": ex.instruction, "gold": gold, "pred": pred, "f1": round(f1, 3)}
            )

    n = len(examples) or 1
    return EvalResult(
        n=len(examples),
        exact_match=exact / n,
        token_f1=f1_total / n,
        samples=samples,
    )


def evaluate_file(
    data_path: str | Path,
    llm: LocalLLM | None = None,
    config: HarnessConfig | None = None,
) -> EvalResult:
    return evaluate(read_jsonl(data_path), llm=llm, config=config)
