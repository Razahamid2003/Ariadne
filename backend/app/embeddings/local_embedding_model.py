"""Local embedding model adapter.

Purpose
-------
Turns chunk text into vector embeddings using a local sentence-transformers model,
which is what makes semantic (meaning-based) retrieval possible.

What it does
------------
Wraps the embedding model behind a small interface and returns normalized embedding
batches for a list of texts.

Flow
----
The model is loaded once on first use, then ``encode()`` converts text batches into
vectors for indexing at build time and for the query at search time.
"""

from dataclasses import dataclass

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.app.core.config import EmbeddingsConfig


@dataclass(frozen=True)
class EmbeddingBatch:
    """
    Normalized embedding batch output.
    """

    vectors: np.ndarray
    dimension: int
    count: int


class SentenceTransformersEmbeddingModel:
    """
    Local SentenceTransformers embedding model.
    """

    def __init__(self, config: EmbeddingsConfig):
        self.config = config
        self.model = SentenceTransformer(
            config.model_name_or_path,
            device=None if config.device == "auto" else config.device,
        )

    def encode(self, texts: list[str], show_progress_bar: bool = True) -> EmbeddingBatch:
        """
        Encode text chunks into normalized vectors.

        Args:
            texts:
                List of chunk texts.

            show_progress_bar:
                Whether to display SentenceTransformers progress.

        Returns:
            EmbeddingBatch:
                Float32 normalized vectors.
        """

        if not texts:
            return EmbeddingBatch(
                vectors=np.empty((0, 0), dtype=np.float32),
                dimension=0,
                count=0,
            )

        vectors = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            show_progress_bar=show_progress_bar,
        )

        vectors = np.asarray(vectors, dtype=np.float32)

        return EmbeddingBatch(
            vectors=vectors,
            dimension=int(vectors.shape[1]),
            count=int(vectors.shape[0]),
        )