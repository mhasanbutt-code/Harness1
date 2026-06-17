"""LoRA / QLoRA fine-tuning for the local LLM.

`train_lora(config, data_path)` runs a real PEFT fine-tune when the optional
`llm` extras are installed. If they're missing it raises a clear, actionable
error instead of pretending to train — training is the one place a silent
fallback would be misleading.

Design notes:
* QLoRA: base weights are loaded in 4-bit (if bitsandbytes is present) and only
  the LoRA adapters are trained, so this fits on a single modest GPU.
* The adapter is saved to ``config.adapter_dir`` and is what `LocalLLM.load`
  picks up automatically afterwards.
"""

from __future__ import annotations

from pathlib import Path

from harness.config import HarnessConfig
from harness.llm.data import build_hf_dataset, read_jsonl


class MissingTrainingDeps(RuntimeError):
    """Raised when the optional training stack isn't installed."""


def train_lora(config: HarnessConfig, data_path: str | Path) -> Path:
    """Fine-tune ``config.base_model`` with LoRA and save the adapter.

    Returns the path to the saved adapter directory.
    """
    try:
        import torch  # noqa: F401
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )
    except Exception as e:  # pragma: no cover - exercised only without extras
        raise MissingTrainingDeps(
            "Training requires the 'llm' extras. Install with:\n"
            "    pip install -e \".[llm]\"\n"
            f"(import failed: {e})"
        ) from e

    examples = read_jsonl(data_path)
    if not examples:
        raise ValueError(f"no training examples found in {data_path}")

    tokenizer = AutoTokenizer.from_pretrained(config.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs: dict = {"torch_dtype": "auto"}
    if config.lora.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype="bfloat16",
                bnb_4bit_use_double_quant=True,
            )
        except Exception:
            pass  # no bitsandbytes -> full-precision LoRA

    model = AutoModelForCausalLM.from_pretrained(config.base_model, **model_kwargs)
    if config.lora.load_in_4bit and "quantization_config" in model_kwargs:
        model = prepare_model_for_kbit_training(model)

    peft_cfg = LoraConfig(
        r=config.lora.r,
        lora_alpha=config.lora.alpha,
        lora_dropout=config.lora.dropout,
        target_modules=config.lora.target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    model.print_trainable_parameters()

    dataset = build_hf_dataset(examples, tokenizer, config.max_seq_len)
    collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

    config.output_dir.mkdir(parents=True, exist_ok=True)
    args = TrainingArguments(
        output_dir=str(config.output_dir),
        per_device_train_batch_size=config.batch_size,
        gradient_accumulation_steps=config.grad_accum,
        num_train_epochs=config.num_epochs,
        learning_rate=config.learning_rate,
        logging_steps=10,
        save_strategy="epoch",
        report_to=[],
        bf16=True,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()

    config.adapter_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(config.adapter_dir))
    tokenizer.save_pretrained(str(config.adapter_dir))
    return config.adapter_dir
