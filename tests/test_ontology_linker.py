"""Unit and benchmark tests for bio-peft-ontology-linker."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from bio_peft.benchmark_strategies import run_benchmark
from bio_peft.dataset_prep import format_chatml_prompt, load_and_validate_sft_jsonl, split_dataset
from bio_peft.ontology_linker import OntologyLinker
from bio_peft.train_lora import run_training


def test_sft_dataset_validation_and_chatml():
    examples, report = load_and_validate_sft_jsonl(ROOT / "data" / "train_sft_bio_ner.jsonl")
    assert report.total_valid == 10
    assert report.duplicates_removed == 0
    train_ex, val_ex = split_dataset(examples, val_ratio=0.2, seed=42)
    assert len(train_ex) == 8
    assert len(val_ex) == 2

    chat = format_chatml_prompt(train_ex[0])
    assert [m["role"] for m in chat["messages"]] == ["system", "user", "assistant"]


def test_lora_training_dry_run_reproducible(tmp_path):
    summary = run_training(ROOT / "configs" / "lora_sft_config.yaml", dry_run=True)
    assert summary["status"] == "dry_run_validated"
    assert summary["lora_r"] == 16
    assert summary["dataset_report"]["train_count"] == 8


def test_ontology_linker_extract_and_disambiguate():
    linker = OntologyLinker(ROOT / "data" / "ontology_catalog.json")
    linked = linker.link(
        "secreted lipid bilayer vesicles from myotubes",
        context="Isolated via 100,000 x g ultracentrifugation.",
        strategy="hybrid_disambiguation",
    )
    assert linked.concept_id == "GO:1903561"

    entities = linker.extract_and_link_text(
        "Dexamethasone induced muscle wasting and upregulated MuRF1."
    )
    ids = {e.concept_id for e in entities}
    assert "CHEBI:41879" in ids
    assert "MESH:D009133" in ids
    assert "HGNC:16009" in ids


def test_hybrid_outperforms_lexical_baseline():
    rows = run_benchmark(
        ROOT / "data" / "ontology_catalog.json",
        ROOT / "data" / "eval_entity_linking.json",
    )
    by_strat = {r["strategy"]: r for r in rows}
    assert by_strat["hybrid_disambiguation"]["top1_accuracy"] >= 0.85
    assert (
        by_strat["hybrid_disambiguation"]["top1_accuracy"]
        > by_strat["exact_lexical"]["top1_accuracy"]
    )
