import asyncio
import json
import os
from collections.abc import Awaitable
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

import httpx
import numpy as np
import pytest
from numpy.typing import NDArray

from test_automation_sdk_mcp.build_index import BuildResult, build_index
from test_automation_sdk_mcp.config import EmbeddingProviderKind, RuntimeConfig
from test_automation_sdk_mcp.documents import EMBEDDING_DIMENSION
from test_automation_sdk_mcp.errors import ConfigurationError
from test_automation_sdk_mcp.index import load_packaged_artifacts, load_verified_artifacts
from test_automation_sdk_mcp.provider.openai import OpenAIEmbeddingProvider
from test_automation_sdk_mcp.server import DocumentationRetriever, create_server

DEFAULT_OPENAI_ENDPOINT = "http://localhost:8080/v1/embeddings"
DEFAULT_OPENAI_MODEL = "nomic-embed-text-v1.5.Q8_0.gguf"
LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


@dataclass(frozen=True, slots=True)
class OpenAIIntegrationSettings:
    endpoint_url: str
    model: str
    api_key: str | None


def run[ResultT](coroutine: Awaitable[ResultT]) -> ResultT:
    return asyncio.run(coroutine)


async def endpoint_is_reachable(endpoint_url: str) -> bool:
    parsed = urlsplit(endpoint_url)
    origin = f"{parsed.scheme}://{parsed.netloc}"
    try:
        async with httpx.AsyncClient(timeout=1.0) as client:
            await client.get(origin)
    except httpx.RequestError:
        return False
    return True


@pytest.fixture(scope="module")
def local_openai() -> OpenAIIntegrationSettings:
    endpoint_url = os.environ.get("TA_SDK_OPENAI_URL", DEFAULT_OPENAI_ENDPOINT)
    parsed = urlsplit(endpoint_url.strip().rstrip("/"))
    if parsed.hostname not in LOCAL_HOSTS:
        pytest.skip("OpenAI integration tests require a loopback TA_SDK_OPENAI_URL.")
    model = os.environ.get("TA_SDK_OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    api_key = os.environ.get("TA_SDK_OPENAI_API_KEY") or None
    try:
        config = RuntimeConfig(
            provider=EmbeddingProviderKind.OPENAI,
            endpoint_url=endpoint_url,
            model=model,
            api_key=api_key,
        )
    except ConfigurationError as error:
        pytest.skip(f"OpenAI integration configuration is invalid: {error.safe_message}")
    if not run(endpoint_is_reachable(config.endpoint_url)):
        pytest.skip("Local OpenAI-compatible endpoint is unavailable.")
    return OpenAIIntegrationSettings(config.endpoint_url, model, api_key)


def make_source(root: Path) -> Path:
    source = root / "source"
    source.mkdir()
    (source / "index.html").write_text("<html><body><h1>Intro</h1><p>Body</p></body></html>", encoding="utf-8")
    (source / "search.json").write_text(
        json.dumps(
            {
                "items": [
                    {
                        "location": "index.html",
                        "level": 1,
                        "title": "Intro",
                        "text": "<p>Body</p>",
                        "path": [],
                        "tags": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return source


def make_provider(settings: OpenAIIntegrationSettings, model: str | None) -> OpenAIEmbeddingProvider:
    return OpenAIEmbeddingProvider(
        settings.endpoint_url,
        model,
        api_key=settings.api_key,
        connect_timeout=2.0,
        request_timeout=30.0,
    )


@pytest.mark.openai
def test_local_openai_returns_finite_768_dimensional_embeddings(
    local_openai: OpenAIIntegrationSettings,
) -> None:
    async def exercise() -> NDArray[np.float32]:
        async with make_provider(local_openai, local_openai.model) as provider:
            return await provider.embed(["A test automation SDK documentation query.", "How do I capture data?"])

    result = run(exercise())

    assert result.shape == (2, EMBEDDING_DIMENSION)
    assert result.dtype == np.dtype(np.float32)
    assert result.flags.c_contiguous
    assert np.isfinite(result).all()


@pytest.mark.openai
def test_openai_builds_model_bound_and_trust_mode_artifacts_and_retrieves(
    local_openai: OpenAIIntegrationSettings,
    tmp_path: Path,
) -> None:
    source = make_source(tmp_path)
    model_bound_output = tmp_path / "model-bound"
    trust_mode_output = tmp_path / "trust-mode"

    async def exercise() -> tuple[BuildResult, BuildResult]:
        model_provider = make_provider(local_openai, local_openai.model)
        try:
            model_bound = await build_index(
                source,
                model_bound_output,
                model_provider,
                model=local_openai.model,
                provider_kind=EmbeddingProviderKind.OPENAI,
            )
        finally:
            await model_provider.aclose()

        trust_provider = make_provider(local_openai, None)
        try:
            trust_mode = await build_index(
                source,
                trust_mode_output,
                trust_provider,
                model=None,
                provider_kind=EmbeddingProviderKind.OPENAI,
            )
        finally:
            await trust_provider.aclose()
        return model_bound, trust_mode

    model_bound_result, trust_mode_result = run(exercise())
    assert model_bound_result.manifest.embedding_provider == "openai"
    assert model_bound_result.manifest.embedding_model == local_openai.model
    assert trust_mode_result.manifest.embedding_provider == "openai"
    assert trust_mode_result.manifest.embedding_model is None

    async def retrieve(output: Path, model: str | None) -> list[str]:
        artifacts = load_verified_artifacts(output)
        provider = make_provider(local_openai, model)
        try:
            retriever = DocumentationRetriever(artifacts, provider, result_count=1)
            return [result.location for result in await retriever.retrieve("What is the introduction?")]
        finally:
            await provider.aclose()

    assert run(retrieve(model_bound_output, local_openai.model)) == ["index.html"]
    assert run(retrieve(trust_mode_output, None)) == ["index.html"]


@pytest.mark.openai
def test_openai_runtime_rejects_packaged_ollama_artifacts(
    local_openai: OpenAIIntegrationSettings,
) -> None:
    config = RuntimeConfig(
        provider=EmbeddingProviderKind.OPENAI,
        endpoint_url=local_openai.endpoint_url,
        model=local_openai.model,
    )

    with pytest.raises(ConfigurationError, match="provider"):
        create_server(config, artifacts=load_packaged_artifacts())
