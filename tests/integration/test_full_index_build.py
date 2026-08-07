import asyncio
from pathlib import Path

import pytest

from test_automation_sdk_mcp.build_index import build_index
from test_automation_sdk_mcp.config import DEFAULT_ENDPOINT_URL, DEFAULT_MODEL, EmbeddingProviderKind
from test_automation_sdk_mcp.index import load_verified_artifacts
from test_automation_sdk_mcp.ollama import OllamaEmbeddingProvider
from test_automation_sdk_mcp.provider import EmbeddingProvenance


@pytest.mark.ollama
def test_full_documentation_index_builds_and_reloads(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    source = repository_root / "data"
    output = tmp_path / "db"

    async def exercise() -> tuple[int, int]:
        async with OllamaEmbeddingProvider(DEFAULT_ENDPOINT_URL, DEFAULT_MODEL) as provider:
            result = await build_index(
                source,
                output,
                provider,
                EmbeddingProvenance(EmbeddingProviderKind.OLLAMA, DEFAULT_MODEL),
            )
        artifacts = load_verified_artifacts(output)
        assert result.source_sections == 846
        assert result.chunk_count == 972
        assert artifacts.index.__class__.__name__ == "IndexFlatL2"
        assert artifacts.index.d == 768
        assert artifacts.index.ntotal == len(artifacts.documents.documents) == 972
        assert artifacts.manifest.embedding_model == DEFAULT_MODEL
        return artifacts.index.ntotal, artifacts.index.d

    vector_count, dimension = asyncio.run(exercise())

    assert (vector_count, dimension) == (972, 768)
