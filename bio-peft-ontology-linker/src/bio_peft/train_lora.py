#!/usr/bin/env python3
"""Reproducible PyTorch + HuggingFace PEFT (LoRA / QLoRA) fine-tuning pipeline.

Supports:
- Full multi-GPU / single-GPU QLoRA fine-tuning via PyTorch, Transformers, PEFT, BitsAndBytes, and TRL.
- `--dry-run` mode to validate YAML configs, dataset schemas, train/val splits, and ChatML formatting
  deterministically in CPU/CI environments prior to GPU job submission.
"""

import argparse
import json
import random
from pathlib import Path

import yaml

from .dataset_prep import format_chatml_prompt, load_and_validate_sft_jsonl, split_dataset


def set_deterministic_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def run_training(config_path: Path, dry_run: bool = False) -> dict:
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    seed = int(cfg["experiment"].get("seed", 42))
    set_deterministic_seed(seed)

    root = Path(config_path).resolve().parents[1]
    data_path = root / cfg["data"]["train_jsonl"]
    examples, report = load_and_validate_sft_jsonl(data_path)
    train_ex, val_ex = split_dataset(
        examples,
        val_ratio=float(cfg["data"].get("validation_split", 0.2)),
        seed=seed,
    )
    report.train_count = len(train_ex)
    report.val_count = len(val_ex)

    train_records = [format_chatml_prompt(e) for e in train_ex]
    val_records = [format_chatml_prompt(e) for e in val_ex]

    output_dir = root / cfg["experiment"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    # Persist frozen experiment configuration and data audit for full reproducibility
    (output_dir / "resolved_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    (output_dir / "dataset_report.json").write_text(report.model_dump_json(indent=2), encoding="utf-8")

    summary = {
        "experiment": cfg["experiment"]["name"],
        "base_model": cfg["model"]["base_model_name"],
        "lora_r": cfg["lora"]["r"],
        "lora_alpha": cfg["lora"]["lora_alpha"],
        "dry_run": dry_run,
        "dataset_report": report.model_dump(),
        "output_dir": str(output_dir),
    }

    if dry_run:
        summary["status"] = "dry_run_validated"
        (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        return summary

    # Live GPU fine-tuning path (PyTorch + HuggingFace Transformers + PEFT + TRL)
    import torch
    from datasets import Dataset
    from peft import LoraConfig, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    model_cfg = cfg["model"]
    lora_cfg = cfg["lora"]
    train_cfg = cfg["training"]

    quant_config = None
    if model_cfg.get("use_4bit_qlora", True) and torch.cuda.is_available():
        compute_dtype = getattr(torch, model_cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=model_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=model_cfg.get("use_double_quant", True),
        )

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model_name"], use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["base_model_name"],
        quantization_config=quant_config,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    )
    if quant_config is not None:
        model = prepare_model_for_kbit_training(model)

    peft_config = LoraConfig(
        r=int(lora_cfg["r"]),
        lora_alpha=int(lora_cfg["lora_alpha"]),
        lora_dropout=float(lora_cfg["lora_dropout"]),
        bias=lora_cfg.get("bias", "none"),
        task_type=lora_cfg.get("task_type", "CAUSAL_LM"),
        target_modules=list(lora_cfg["target_modules"]),
    )

    train_ds = Dataset.from_list(train_records)
    val_ds = Dataset.from_list(val_records)

    sft_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=float(train_cfg["num_train_epochs"]),
        per_device_train_batch_size=int(train_cfg["per_device_train_batch_size"]),
        gradient_accumulation_steps=int(train_cfg["gradient_accumulation_steps"]),
        learning_rate=float(train_cfg["learning_rate"]),
        lr_scheduler_type=train_cfg.get("lr_scheduler_type", "cosine"),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.05)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        bf16=bool(train_cfg.get("bf16", True) and torch.cuda.is_available()),
        gradient_checkpointing=bool(train_cfg.get("gradient_checkpointing", True)),
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        eval_strategy=train_cfg.get("eval_strategy", "epoch"),
        save_strategy=train_cfg.get("save_strategy", "epoch"),
        max_length=int(cfg["data"].get("max_seq_length", 1024)),
        seed=seed,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    train_result = trainer.train()
    trainer.save_model(str(output_dir / "lora_adapter"))
    tokenizer.save_pretrained(str(output_dir / "lora_adapter"))

    summary["status"] = "trained"
    summary["train_loss"] = float(train_result.training_loss)
    (output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible PEFT/LoRA biomedical fine-tuning.")
    parser.add_argument("--config", type=Path, default=Path("configs/lora_sft_config.yaml"))
    parser.add_argument("--dry-run", action="store_true", help="Validate config, schema, and splits without GPU.")
    args = parser.parse_args()
    result = run_training(args.config, dry_run=args.dry_run)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
