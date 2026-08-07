"""Embedding provider implementations and their shared contract."""

from .embeddings import (
    DEFAULT_BATCH_SIZE,
    EMBEDDING_DIMENSION,
    EmbeddingProvenance,
    EmbeddingProvider,
)
from .factory import create_embedding_provider
from .ollama import OllamaEmbeddingProvider
from .openai import OpenAIEmbeddingProvider

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "EMBEDDING_DIMENSION",
    "EmbeddingProvenance",
    "EmbeddingProvider",
    "OllamaEmbeddingProvider",
    "OpenAIEmbeddingProvider",
    "create_embedding_provider",
]
