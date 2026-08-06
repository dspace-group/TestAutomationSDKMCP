import asyncio

import pytest

from test_automation_sdk_mcp.config import DEFAULT_ENDPOINT_URL, DEFAULT_MODEL
from test_automation_sdk_mcp.index import load_packaged_artifacts
from test_automation_sdk_mcp.ollama import OllamaEmbeddingProvider
from test_automation_sdk_mcp.server import DocumentationRetriever


@pytest.mark.ollama
def test_packaged_index_retrieves_representative_documentation_locations() -> None:
    async def exercise() -> dict[str, list[str]]:
        artifacts = load_packaged_artifacts()
        async with OllamaEmbeddingProvider(DEFAULT_ENDPOINT_URL, DEFAULT_MODEL) as provider:
            retriever = DocumentationRetriever(artifacts, provider, result_count=5)
            queries = {
                "capturing": "How do I capture a TestAutomationSDK experiment?",
                "test-environment": "How can a test environment be accessed?",
                "scenario": "What is a scenario and how is it configured?",
            }
            return {
                name: [result.location for result in await retriever.retrieve(query)] for name, query in queries.items()
            }

    locations = asyncio.run(exercise())

    assert any(location.startswith("concepts/capture.html") for location in locations["capturing"])
    assert any(
        location.startswith("concepts/test-environment-access.html") for location in locations["test-environment"]
    )
    assert any(location.startswith("concepts/scenario.html") for location in locations["scenario"])
