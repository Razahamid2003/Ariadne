"""Optional local reranking.

Purpose
-------
Re-scores the top retrieved candidates with a local cross-encoder so the most
relevant evidence rises to the top for the model.

What it does
------------
When enabled and a model is available, it scores each candidate against the query
and reorders the pool; otherwise retrieval proceeds unchanged.

Flow
----
The candidate pool and query are scored together, the scores are normalized, and
the candidates are reordered before answer generation.
"""

from __future__ import annotations

from typing import Any

from backend.app.core.config import Settings
from backend.app.retrieval.models import RetrievalCandidate


class LocalReranker:
    """Rerank retrieved chunks with a local sentence-transformers cross-encoder."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._model: Any | None = None
        self._load_error: str | None = None

    def rerank(self, query: str, candidates: list[RetrievalCandidate], top_k: int) -> tuple[list[RetrievalCandidate], dict[str, Any]]:
        cfg = self.settings.retrieval
        if not getattr(cfg, "rerank_enabled", False):
            return candidates[:top_k], {"reranker_enabled": False}

        if not candidates:
            return [], {"reranker_enabled": True, "reranker_used": False, "reason": "no candidates"}

        model = self._load_model()
        if model is None:
            return candidates[:top_k], {
                "reranker_enabled": True,
                "reranker_used": False,
                "reranker_error": self._load_error or "model unavailable",
            }

        pool_size = max(top_k, min(int(getattr(cfg, "rerank_candidate_pool", 30)), len(candidates)))
        pool = candidates[:pool_size]
        pairs = [(query, self._candidate_text(item)) for item in pool]

        try:
            scores = model.predict(pairs, show_progress_bar=False)
        except TypeError:
            scores = model.predict(pairs)
        except Exception as exc:  # pragma: no cover - depends on local model runtime
            return candidates[:top_k], {
                "reranker_enabled": True,
                "reranker_used": False,
                "reranker_error": str(exc),
            }

        raw_scores = [float(score) for score in list(scores)]
        normalized = self._normalize_scores(raw_scores)
        base_scores = [item.combined_score for item in pool]
        base_normalized = self._normalize_scores(base_scores)

        for item, raw, norm, base_norm in zip(pool, raw_scores, normalized, base_normalized):
            item.reranker_score = raw
            item.combined_score = 0.80 * norm + 0.20 * base_norm
            item.add_reason("reranker")

        ranked_pool = sorted(pool, key=lambda item: item.combined_score, reverse=True)
        ranked = ranked_pool + candidates[pool_size:]
        return ranked[:top_k], {
            "reranker_enabled": True,
            "reranker_used": True,
            "reranker_model": getattr(cfg, "rerank_model_name_or_path", ""),
            "reranker_pool": pool_size,
            "reranker_returned": min(top_k, len(ranked)),
        }

    def _load_model(self):
        if self._model is not None:
            return self._model
        if self._load_error:
            return None
        try:
            from sentence_transformers import CrossEncoder

            model_name = getattr(self.settings.retrieval, "rerank_model_name_or_path", "cross-encoder/ms-marco-MiniLM-L-6-v2")
            device = getattr(self.settings.retrieval, "rerank_device", "auto")
            kwargs = {}
            if device == "cuda":
                try:
                    import torch

                    if not torch.cuda.is_available():
                        device = "cpu"
                except Exception:
                    device = "cpu"
            if device in {"cpu", "cuda"}:
                kwargs["device"] = device
            self._model = CrossEncoder(model_name, **kwargs)
            return self._model
        except Exception as exc:  # pragma: no cover - depends on local environment
            self._load_error = str(exc)
            return None

    @staticmethod
    def _candidate_text(candidate: RetrievalCandidate) -> str:
        title = candidate.title or ""
        source = candidate.source_file or ""
        text = candidate.text or ""
        return f"{title}\n{source}\n{text}"[:3000]

    @staticmethod
    def _normalize_scores(values: list[float]) -> list[float]:
        if not values:
            return []
        lo = min(values)
        hi = max(values)
        if hi == lo:
            return [1.0 for _ in values]
        return [(value - lo) / (hi - lo) for value in values]
