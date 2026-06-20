"""Embedding model adapters.

Purpose
-------
Holds the local embedding model used for semantic search. Embedding generation is
kept separate from answer generation, so the answer model can change without
re-embedding the corpus.
"""
