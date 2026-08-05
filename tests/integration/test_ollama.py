import asyncio

import numpy as np
import pytest

from test_automation_sdk_mcp.config import DEFAULT_ENDPOINT_URL, DEFAULT_MODEL
from test_automation_sdk_mcp.ollama import EMBEDDING_DIMENSION, OllamaEmbeddingProvider


@pytest.mark.ollama
def test_local_ollama_returns_finite_768_dimensional_embedding() -> None:
    async def exercise() -> np.ndarray:
        async with OllamaEmbeddingProvider(DEFAULT_ENDPOINT_URL, DEFAULT_MODEL) as provider:
            return await provider.embed(["A test automation SDK documentation query."])

    result = asyncio.run(exercise())

    assert result.shape == (1, EMBEDDING_DIMENSION)
    assert result.dtype == np.dtype(np.float32)
    assert np.isfinite(result).all()
