"""FAISS-based hybrid retrieval with keyword boosting."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import faiss
import numpy as np

from app.catalog import Assessment
from app.config import FAISS_INDEX_PATH, TOP_K_RETRIEVAL
from app.embeddings import EmbeddingModel

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """Hybrid retrieval combining FAISS semantic search with keyword boosting."""

    def __init__(self, assessments: list[Assessment]):
        self.assessments = assessments
        self._name_lower_map = {a.name.lower(): i for i, a in enumerate(assessments)}
        self._keyword_index: dict[str, set[int]] = {}

        logger.info("Loading embedding model")
        self.model = EmbeddingModel()

        index_path = Path(FAISS_INDEX_PATH)
        embeddings_path = index_path.parent / "embeddings.npy"

        if index_path.exists() and embeddings_path.exists():
            cached = np.load(embeddings_path)
            if cached.shape[0] == len(assessments):
                logger.info("Loading cached FAISS index from %s", index_path)
                self.index = faiss.read_index(str(index_path))
                self.embeddings = cached.astype(np.float32)
            else:
                self._build_and_cache_index(assessments, index_path, embeddings_path)
        else:
            self._build_and_cache_index(assessments, index_path, embeddings_path)

        self._build_keyword_index()
        logger.info("Retrieval engine ready with %d assessments", len(assessments))

    def _build_and_cache_index(
        self,
        assessments: list[Assessment],
        index_path: Path,
        embeddings_path: Path,
    ) -> None:
        search_texts = [a.search_text for a in assessments]
        logger.info("Encoding %d assessments...", len(search_texts))
        embeddings = self.model.encode(search_texts, normalize_embeddings=True)
        self.embeddings = embeddings

        dim = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(self.embeddings)

        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_path))
        np.save(embeddings_path, self.embeddings)
        logger.info("Saved FAISS index to %s", index_path)

    def _build_keyword_index(self) -> None:
        for i, assessment in enumerate(self.assessments):
            text = f"{assessment.name} {assessment.description}".lower()
            for key in assessment.keys:
                text += f" {key.lower()}"
            for level in assessment.job_levels:
                text += f" {level.lower()}"

            words = set(re.findall(r"[a-z0-9#+.]+", text))
            for word in words:
                self._keyword_index.setdefault(word, set()).add(i)

    def search(
        self,
        query: str,
        top_k: int | None = None,
        job_level_filter: str | None = None,
        type_filter: list[str] | None = None,
    ) -> list[tuple[Assessment, float]]:
        k = top_k or TOP_K_RETRIEVAL

        query_vec = self.model.encode([query], normalize_embeddings=True)

        n_search = min(len(self.assessments), k * 3)
        scores, indices = self.index.search(query_vec, n_search)

        results: dict[int, float] = {}
        for idx, score in zip(indices[0], scores[0], strict=False):
            if idx < 0:
                continue
            results[idx] = float(score)

        query_words = set(re.findall(r"[a-z0-9#+.]+", query.lower()))
        for word in query_words:
            for idx in self._keyword_index.get(word, set()):
                if idx in results:
                    results[idx] += 0.15
                else:
                    results[idx] = 0.3

        filtered_results: list[tuple[int, float]] = []
        for idx, score in results.items():
            assessment = self.assessments[idx]

            if job_level_filter:
                level_lower = job_level_filter.lower()
                levels = [lvl.lower() for lvl in assessment.job_levels]
                if not any(level_lower in lvl or lvl in level_lower for lvl in levels):
                    score *= 0.7

            if type_filter:
                type_match = any(tf.lower() in [key.lower() for key in assessment.keys] for tf in type_filter)
                if not type_match:
                    score *= 0.6

            filtered_results.append((idx, score))

        filtered_results.sort(key=lambda item: item[1], reverse=True)
        return [(self.assessments[idx], score) for idx, score in filtered_results[:k]]

    def search_by_names(self, names: list[str]) -> list[Assessment]:
        results: list[Assessment] = []
        for name in names:
            name_lower = name.lower().strip()
            if name_lower in self._name_lower_map:
                results.append(self.assessments[self._name_lower_map[name_lower]])
                continue
            for catalog_name, idx in self._name_lower_map.items():
                if name_lower in catalog_name or catalog_name in name_lower:
                    results.append(self.assessments[idx])
                    break
        return results
