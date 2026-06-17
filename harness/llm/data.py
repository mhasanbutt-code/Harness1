"""Instruction dataset preparation.

Reads JSONL where each line is an example with ``instruction`` and ``response``
(plus optional ``input`` context). Produces formatted prompt/target strings and,
when `transformers` is available, a tokenized `datasets.Dataset` ready for the
LoRA trainer. Without those deps it still yields the formatted records, which is
enough for the eval loop and tests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

PROMPT_TEMPLATE = (
    "### Instruction:\n{instruction}\n"
    "{input_block}"
    "### Response:\n"
)


@dataclass
class Example:
    instruction: str
    response: str
    input: str = ""

    def prompt(self) -> str:
        input_block = f"### Input:\n{self.input}\n" if self.input.strip() else ""
        return PROMPT_TEMPLATE.format(instruction=self.instruction, input_block=input_block)

    def full_text(self, eos: str = "") -> str:
        return self.prompt() + self.response + eos


def read_jsonl(path: str | Path) -> list[Example]:
    """Load and validate examples from a JSONL file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"training data not found: {path}")
    examples: list[Example] = []
    for lineno, line in enumerate(path.read_text().splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj: dict[str, Any] = json.loads(line)
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{lineno}: invalid JSON ({e})") from e
        if "instruction" not in obj or "response" not in obj:
            raise ValueError(f"{path}:{lineno}: needs 'instruction' and 'response'")
        examples.append(
            Example(
                instruction=str(obj["instruction"]),
                response=str(obj["response"]),
                input=str(obj.get("input", "")),
            )
        )
    return examples


def iter_formatted(examples: list[Example], eos: str = "") -> Iterator[str]:
    for ex in examples:
        yield ex.full_text(eos)


def build_hf_dataset(examples: list[Example], tokenizer, max_seq_len: int):
    """Tokenize examples into a `datasets.Dataset` for causal-LM training.

    Loss is computed over the whole sequence (prompt + response). For
    response-only loss, mask prompt tokens in a custom collator; kept simple
    here on purpose.
    """
    from datasets import Dataset

    eos = tokenizer.eos_token or ""
    texts = [ex.full_text(eos) for ex in examples]

    def _tokenize(batch: dict[str, list[str]]) -> dict[str, Any]:
        enc = tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_len,
            padding=False,
        )
        enc["labels"] = [ids.copy() for ids in enc["input_ids"]]
        return enc

    ds = Dataset.from_dict({"text": texts})
    return ds.map(_tokenize, batched=True, remove_columns=["text"])
