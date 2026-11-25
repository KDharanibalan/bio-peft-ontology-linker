"""Two-stage Biomedical Entity Linking & Ontology Normalization Engine.

Stage 1: Vector candidate retrieval across MeSH / GO / ChEBI / HGNC / ICD-10 concepts.
Stage 2: Context-aware disambiguation (lexical+definition reranker or fine-tuned LLM / API).
"""

import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer


class OntologyConcept(BaseModel):
    concept_id: str
    ontology: str
    canonical_name: str
    category: str
    synonyms: List[str]
    definition: str

    def searchable_document(self) -> str:
        syns = " ; ".join(self.synonyms)
        return f"{self.concept_id} {self.canonical_name} {self.category} {syns} {self.definition}"


class LinkedEntity(BaseModel):
    mention: str
    concept_id: str
    canonical_name: str
    ontology: str
    score: float = Field(ge=0.0, le=1.0)
    strategy: str
    top_candidates: List[Tuple[str, float]]


def _norm_tokens(text: str) -> set:
    return set(re.findall(r"[a-z0-9][a-z0-9\-\.]+", text.lower()))


class OntologyLinker:
    """Normalizes noisy biological/clinical free text to standardized ontology identifiers."""

    def __init__(self, catalog_path: Path):
        raw = json.loads(Path(catalog_path).read_text(encoding="utf-8"))
        self.concepts: List[OntologyConcept] = [OntologyConcept.model_validate(item) for item in raw]
        self._concept_by_id = {c.concept_id: c for c in self.concepts}

        # Character + word TF-IDF vector index handles biological spelling variants and acronyms
        docs = [c.searchable_document() for c in self.concepts]
        self._word_vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, lowercase=True)
        self._char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, lowercase=True)
        self._word_mat = self._word_vec.fit_transform(docs).toarray().astype(np.float32)
        self._char_mat = self._char_vec.fit_transform(docs).toarray().astype(np.float32)

    def _lexical_scores(self, mention: str) -> np.ndarray:
        """Strict dictionary/alias lookup baseline (canonical name, concept ID, or synonym substring)."""
        m_low = mention.lower().strip()
        scores = np.zeros(len(self.concepts), dtype=np.float64)
        for i, c in enumerate(self.concepts):
            aliases = [c.canonical_name.lower(), c.concept_id.lower()] + [s.lower() for s in c.synonyms]
            if any(a == m_low or (len(a) >= 4 and a in m_low) for a in aliases):
                scores[i] = 1.0
        return scores

    def _vector_scores(self, text: str) -> np.ndarray:
        wv = self._word_vec.transform([text]).toarray().astype(np.float32)[0]
        cv = self._char_vec.transform([text]).toarray().astype(np.float32)[0]
        ws = self._word_mat @ wv
        cs = self._char_mat @ cv
        return (0.55 * ws + 0.45 * cs).astype(np.float64)

    def _context_rerank(
        self,
        mention: str,
        context: str,
        candidate_indices: List[int],
        base_scores: np.ndarray,
    ) -> List[Tuple[int, float]]:
        """Stage-2 context disambiguation anchoring on mention identity while using context to break ties."""
        ctx_vec_scores = self._vector_scores(f"{mention}. {context}") if context else base_scores
        mention_lex = self._lexical_scores(mention)
        ctx_toks = _norm_tokens(f"{mention} {context}")

        reranked: List[Tuple[int, float]] = []
        for idx in candidate_indices:
            c = self.concepts[idx]
            def_toks = _norm_tokens(f"{c.canonical_name} {' '.join(c.synonyms)} {c.definition}")
            ctx_overlap = len(ctx_toks & def_toks) / max(1, len(ctx_toks))
            code_boost = 0.25 if c.concept_id.split(":")[-1].lower() in f"{mention} {context}".lower() else 0.0
            final = (
                0.42 * float(base_scores[idx])
                + 0.28 * float(ctx_vec_scores[idx])
                + 0.22 * float(mention_lex[idx])
                + 0.08 * ctx_overlap
                + code_boost
            )
            reranked.append((idx, min(1.0, round(final, 4))))
        reranked.sort(key=lambda x: x[1], reverse=True)
        return reranked

    def link(
        self,
        mention: str,
        context: str = "",
        strategy: str = "hybrid_disambiguation",
        top_k: int = 3,
    ) -> LinkedEntity:
        """Link a noisy free-text entity mention to a canonical ontology concept."""
        if strategy == "exact_lexical":
            scores = self._lexical_scores(mention)
            order = np.argsort(-scores)[:top_k]
            ranked = [(int(i), min(1.0, round(float(scores[int(i)]), 4))) for i in order]
        elif strategy == "dense_vector":
            scores = self._vector_scores(mention)
            order = np.argsort(-scores)[:top_k]
            ranked = [(int(i), min(1.0, round(float(scores[int(i)]), 4))) for i in order]
        elif strategy == "hybrid_disambiguation":
            base = self._vector_scores(mention)
            ctx = self._vector_scores(f"{mention}. {context}") if context else base
            combined_pool = np.argsort(-(0.65 * base + 0.35 * ctx))[: min(8, len(self.concepts))]
            ranked = self._context_rerank(mention, context, [int(i) for i in combined_pool], base)[:top_k]
        else:
            raise ValueError(f"Unsupported linking strategy: {strategy}")

        best_idx, best_score = ranked[0]
        best = self.concepts[best_idx]
        return LinkedEntity(
            mention=mention,
            concept_id=best.concept_id,
            canonical_name=best.canonical_name,
            ontology=best.ontology,
            score=best_score,
            strategy=strategy,
            top_candidates=[(self.concepts[i].concept_id, sc) for i, sc in ranked],
        )

    def extract_and_link_text(self, text: str, threshold: float = 0.25) -> List[LinkedEntity]:
        """Scan a biological sentence for ontology concepts and return normalized entity records."""
        low = text.lower()
        found: List[LinkedEntity] = []
        seen = set()
        for c in self.concepts:
            terms = [c.canonical_name] + c.synonyms
            for term in sorted(terms, key=len, reverse=True):
                if len(term) >= 3 and re.search(rf"\b{re.escape(term.lower())}\b", low):
                    if c.concept_id not in seen:
                        linked = self.link(term, context=text, strategy="hybrid_disambiguation")
                        if linked.score >= threshold:
                            found.append(linked)
                            seen.add(c.concept_id)
                    break
        return found
