"""Reproducible SFT dataset validation, deduplication, splitting, and ChatML formatting."""

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

from pydantic import BaseModel, Field


class SFTExample(BaseModel):
    id: str
    task: str
    instruction: str
    input: str
    output: str


class DatasetReport(BaseModel):
    total_raw: int
    total_valid: int
    duplicates_removed: int
    task_distribution: Dict[str, int]
    avg_prompt_words: float
    avg_output_words: float
    train_count: int = 0
    val_count: int = 0


def format_chatml_prompt(example: SFTExample) -> Dict[str, List[Dict[str, str]]]:
    """Convert an instruction-tuning record into ChatML messages for HuggingFace TRL SFTTrainer."""
    return {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a domain-specialized biomedical NLP assistant trained for accurate "
                    "entity normalization, ontology mapping, and scientific QA."
                ),
            },
            {
                "role": "user",
                "content": f"{example.instruction}\n\nInput:\n{example.input}",
            },
            {
                "role": "assistant",
                "content": example.output,
            },
        ]
    }


def load_and_validate_sft_jsonl(path: Path) -> Tuple[List[SFTExample], DatasetReport]:
    """Load, schema-validate, and deduplicate an SFT JSONL file."""
    raw_lines = [ln for ln in Path(path).read_text(encoding="utf-8").splitlines() if ln.strip()]
    seen_inputs = set()
    examples: List[SFTExample] = []
    dups = 0
    task_counts: Dict[str, int] = {}

    for line in raw_lines:
        record = SFTExample.model_validate(json.loads(line))
        dedup_key = (record.task, record.input.strip().lower())
        if dedup_key in seen_inputs:
            dups += 1
            continue
        seen_inputs.add(dedup_key)
        examples.append(record)
        task_counts[record.task] = task_counts.get(record.task, 0) + 1

    n = max(1, len(examples))
    avg_p = round(sum(len((e.instruction + " " + e.input).split()) for e in examples) / n, 2)
    avg_o = round(sum(len(e.output.split()) for e in examples) / n, 2)

    report = DatasetReport(
        total_raw=len(raw_lines),
        total_valid=len(examples),
        duplicates_removed=dups,
        task_distribution=task_counts,
        avg_prompt_words=avg_p,
        avg_output_words=avg_o,
    )
    return examples, report


def split_dataset(
    examples: List[SFTExample],
    val_ratio: float = 0.2,
    seed: int = 42,
) -> Tuple[List[SFTExample], List[SFTExample]]:
    """Deterministic seeded train/validation split."""
    rng = random.Random(seed)
    shuffled = list(examples)
    rng.shuffle(shuffled)
    val_n = max(1, int(round(len(shuffled) * val_ratio)))
    return shuffled[val_n:], shuffled[:val_n]
