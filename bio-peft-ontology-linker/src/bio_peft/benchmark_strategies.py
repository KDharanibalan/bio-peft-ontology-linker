#!/usr/bin/env python3
"""Benchmark exact lexical vs dense vector vs hybrid context-aware ontology normalization."""

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from bio_peft.ontology_linker import OntologyLinker


def run_benchmark(catalog_path: Path, eval_path: Path) -> list:
    linker = OntologyLinker(catalog_path)
    cases = json.loads(Path(eval_path).read_text(encoding="utf-8"))
    strategies = ["exact_lexical", "dense_vector", "hybrid_disambiguation"]

    rows = []
    for strat in strategies:
        top1 = 0
        top3 = 0
        mrr = 0.0
        t0 = time.perf_counter()
        for case in cases:
            res = linker.link(case["mention"], context=case.get("context", ""), strategy=strat, top_k=3)
            gold = case["gold_concept_id"]
            cand_ids = [cid for cid, _ in res.top_candidates]
            if res.concept_id == gold:
                top1 += 1
            if gold in cand_ids:
                top3 += 1
                mrr += 1.0 / (cand_ids.index(gold) + 1)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        n = max(1, len(cases))
        rows.append(
            {
                "strategy": strat,
                "num_cases": len(cases),
                "top1_accuracy": round(top1 / n, 4),
                "top3_recall": round(top3 / n, 4),
                "mrr": round(mrr / n, 4),
                "avg_latency_ms": round(elapsed_ms / n, 3),
            }
        )
    return rows


def main() -> None:
    reports_dir = ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    rows = run_benchmark(ROOT / "data" / "ontology_catalog.json", ROOT / "data" / "eval_entity_linking.json")

    (reports_dir / "ontology_benchmark.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    lines = [
        "# Biomedical Ontology Normalization Benchmark (15 Noisy Mentions)",
        "",
        "| Strategy | Top-1 Accuracy | Top-3 Recall | MRR | Avg Latency (ms) |",
        "| :--- | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| `{r['strategy']}` | {r['top1_accuracy']:.3f} | {r['top3_recall']:.3f} "
            f"| {r['mrr']:.3f} | {r['avg_latency_ms']:.3f} |"
        )
    md = "\n".join(lines) + "\n"
    (reports_dir / "ontology_benchmark.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
