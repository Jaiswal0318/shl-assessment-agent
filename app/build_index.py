"""Build and persist FAISS index and embedding matrix for the catalog."""

from __future__ import annotations

import logging
from pathlib import Path

import faiss
import numpy as np

from app.catalog import load_catalog
from app.config import CATALOG_PATH, FAISS_INDEX_PATH
from app.embeddings import EmbeddingModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EMBEDDINGS_PATH = Path(FAISS_INDEX_PATH).parent / "embeddings.npy"


def build_index() -> None:
    assessments = load_catalog(CATALOG_PATH)
    logger.info("Building index for %d assessments", len(assessments))

    model = EmbeddingModel()
    texts = [a.search_text for a in assessments]
    matrix = model.encode(texts, normalize_embeddings=True)

    dim = matrix.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(matrix)

    index_path = Path(FAISS_INDEX_PATH)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(index_path))
    np.save(EMBEDDINGS_PATH, matrix)

    logger.info("Wrote FAISS index to %s", index_path)
    logger.info("Wrote embeddings to %s", EMBEDDINGS_PATH)


if __name__ == "__main__":
    build_index()
