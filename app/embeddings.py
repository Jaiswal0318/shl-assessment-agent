"""Lightweight embedding encoder using FastEmbed (ONNX, no PyTorch)."""

from __future__ import annotations

import numpy as np
from fastembed import TextEmbedding

from app.config import EMBEDDING_MODEL


class EmbeddingModel:
    """Wraps FastEmbed for normalized vector encoding."""

    def __init__(self, model_name: str | None = None):
        name = model_name or EMBEDDING_MODEL
        self._model = TextEmbedding(model_name=name)

    def encode(self, texts: list[str], normalize_embeddings: bool = True) -> np.ndarray:
        vectors = list(self._model.embed(texts))
        matrix = np.array(vectors, dtype=np.float32)
        if normalize_embeddings:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = matrix / np.clip(norms, 1e-12, None)
        return matrix
