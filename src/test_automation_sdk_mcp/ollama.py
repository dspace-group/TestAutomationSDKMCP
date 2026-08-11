"""Compatibility imports for the legacy Ollama provider module."""

from .provider.ollama import (
    DEFAULT_BATCH_SIZE,
    EMBEDDING_DIMENSION,
    EmbeddingProvider,
    OllamaEmbeddingProvider,
    OllamaEmbeddingRequest,
    OllamaEmbeddingResponse,
)

__all__ = [
    "DEFAULT_BATCH_SIZE",
    "EMBEDDING_DIMENSION",
    "EmbeddingProvider",
    "OllamaEmbeddingProvider",
    "OllamaEmbeddingRequest",
    "OllamaEmbeddingResponse",
]
