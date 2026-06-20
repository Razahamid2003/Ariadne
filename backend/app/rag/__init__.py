"""Answer-generation package.

Purpose
-------
Turns retrieved evidence into grounded, cited answers using the local language
model: context building, prompt construction, generation, and citation validation.
"""

from backend.app.rag.answer_generator import RAGAnswerGenerator
from backend.app.rag.models import RAGAnswerRequest, RAGAnswerResponse

__all__ = ["RAGAnswerGenerator", "RAGAnswerRequest", "RAGAnswerResponse"]
