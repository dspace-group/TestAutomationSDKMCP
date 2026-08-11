"""Composition factory for the selected embedding provider."""

from typing import cast

from ..config import EmbeddingProviderKind, RuntimeConfig
from ..errors import ConfigurationError
from .embeddings import EmbeddingProvider
from .ollama import OllamaEmbeddingProvider
from .openai import OpenAIEmbeddingProvider


def create_embedding_provider(config: RuntimeConfig) -> EmbeddingProvider:
    """Create the configured provider without reading environment state."""

    if config.provider is EmbeddingProviderKind.OLLAMA:
        return OllamaEmbeddingProvider(
            config.endpoint_url,
            cast(str, config.model),
            api_key=config.api_key,
            connect_timeout=config.connect_timeout,
            request_timeout=config.request_timeout,
        )
    if config.provider is EmbeddingProviderKind.OPENAI:
        return OpenAIEmbeddingProvider(
            config.endpoint_url,
            config.model,
            api_key=config.api_key,
            connect_timeout=config.connect_timeout,
            request_timeout=config.request_timeout,
        )
    raise ConfigurationError("Unknown embedding provider.")


__all__ = ["create_embedding_provider"]
