import asyncio
import json
from collections.abc import Awaitable, Sequence
from hashlib import sha256
from pathlib import Path

import faiss
import numpy as np
import pytest
from mcp.client._memory import InMemoryTransport
from mcp.client.session import ClientSession
from mcp.server.mcpserver.exceptions import ToolError
from numpy.typing import NDArray

from test_automation_sdk_mcp.build_index import build_index
from test_automation_sdk_mcp.config import RuntimeConfig
from test_automation_sdk_mcp.documents import ChunkingManifest, DocumentRecord, DocumentStore, IndexManifest
from test_automation_sdk_mcp.errors import ConfigurationError, EmbeddingError, RetrievalError
from test_automation_sdk_mcp.index import ArtifactPaths, LoadedArtifacts
from test_automation_sdk_mcp.server import (
    MAX_QUERY_LENGTH,
    SERVER_INSTRUCTIONS,
    TOOL_DESCRIPTION,
    DocumentationRetriever,
    create_server,
)


class FakeEmbeddingProvider:
    def __init__(self) -> None:
        self.inputs: list[tuple[str, ...]] = []
        self.closed = False

    async def embed(self, inputs: Sequence[str]) -> NDArray[np.float32]:
        self.inputs.append(tuple(inputs))
        return np.zeros((len(inputs), 768), dtype=np.float32)

    async def aclose(self) -> None:
        self.closed = True


class FakeSearchIndex:
    def __init__(self, row_id: int) -> None:
        self.ntotal = 1
        self.row_id = row_id

    def search(self, query: NDArray[np.float32], result_count: int) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.zeros((1, result_count), dtype=np.float32),
            np.full((1, result_count), self.row_id, dtype=np.int64),
        )


def make_artifacts(tmp_path: Path, index: object | None = None, title: str = "First") -> LoadedArtifacts:
    documents = DocumentStore(
        schema_version=1,
        documents=(
            DocumentRecord(
                id="first",
                location="index.html#first",
                title=title,
                breadcrumbs=("Guide",),
                tags=(),
                chunk_index=0,
                content="First content",
            ),
            DocumentRecord(
                id="second",
                location="index.html#second",
                title="Second",
                breadcrumbs=("Reference",),
                tags=(),
                chunk_index=0,
                content="Second content",
            ),
        ),
    )
    digest = "a" * 64
    manifest = IndexManifest(
        schema_version=1,
        index_type="IndexFlatL2",
        distance_metric="l2",
        embedding_provider="ollama",
        embedding_model="fake-model",
        embedding_dimension=768,
        document_count=2,
        search_json_sha256=digest,
        html_tree_sha256=digest,
        faiss_sha256=digest,
        documents_sha256=digest,
        chunking=ChunkingManifest(max_characters=1000, overlap_characters=200),
    )
    if index is None:
        index = faiss.IndexFlatL2(768)
        vectors = np.zeros((2, 768), dtype=np.float32)
        vectors[1, 0] = 3.0
        index.add(vectors)  # pyright: ignore[reportUnknownMemberType]
    return LoadedArtifacts(index, documents, manifest, ArtifactPaths(tmp_path))


def make_retriever(tmp_path: Path, index: object | None = None) -> tuple[DocumentationRetriever, FakeEmbeddingProvider]:
    provider = FakeEmbeddingProvider()
    retriever = DocumentationRetriever(make_artifacts(tmp_path, index), provider, result_count=5)
    return retriever, provider


def run[ResultT](coroutine: Awaitable[ResultT]) -> ResultT:
    return asyncio.run(coroutine)


def test_retrieval_maps_nearest_rows_and_bounds_result_count(tmp_path: Path) -> None:
    retriever, provider = make_retriever(tmp_path)

    results = run(retriever.retrieve("capture documentation"))

    assert [result.location for result in results] == ["index.html#first", "index.html#second"]  # type: ignore[union-attr]
    assert [result.distance for result in results] == [0.0, 9.0]  # type: ignore[union-attr]
    assert provider.inputs == [("capture documentation",)]


@pytest.mark.parametrize("row_id", [-1, 2])
def test_retrieval_rejects_invalid_faiss_row_ids(tmp_path: Path, row_id: int) -> None:
    retriever, _ = make_retriever(tmp_path, FakeSearchIndex(row_id))

    with pytest.raises(RetrievalError, match="invalid row ID"):
        run(retriever.retrieve("query"))


@pytest.mark.parametrize(
    "query",
    ["", "   \t\r\n", "x" * (MAX_QUERY_LENGTH + 1)],
)
def test_retrieval_rejects_invalid_queries(tmp_path: Path, query: str) -> None:
    retriever, _ = make_retriever(tmp_path)

    with pytest.raises(RetrievalError, match="query"):
        run(retriever.retrieve(query))


def test_server_exposes_one_strict_annotated_structured_tool(tmp_path: Path) -> None:
    provider = FakeEmbeddingProvider()
    config = RuntimeConfig(model="fake-model", artifact_directory=tmp_path)
    server = create_server(config, artifacts=make_artifacts(tmp_path), provider=provider)

    tools = run(server.list_tools())
    assert len(tools) == 1  # type: ignore[arg-type]
    tool = tools[0]  # type: ignore[index]
    assert server.instructions == SERVER_INSTRUCTIONS
    assert tool.name == "retrieve_documentation"  # type: ignore[union-attr]
    assert tool.description == TOOL_DESCRIPTION  # type: ignore[union-attr]
    assert tool.input_schema["required"] == ["query"]  # type: ignore[union-attr]
    assert tool.input_schema["properties"]["query"]["type"] == "string"  # type: ignore[union-attr]
    assert "self-contained Test Automation SDK question" in tool.input_schema["properties"]["query"]["description"]  # type: ignore[union-attr]
    assert tool.output_schema is not None  # type: ignore[union-attr]
    assert tool.output_schema["type"] == "object"  # type: ignore[union-attr]
    assert tool.output_schema["properties"]["result"]["type"] == "array"  # type: ignore[union-attr]
    assert (  # type: ignore[union-attr]
        tool.output_schema["properties"]["result"]["items"]["$ref"] == "#/$defs/DocumentationSnippet"
    )
    assert tool.annotations.model_dump(by_alias=True, exclude_none=True) == {  # type: ignore[union-attr]
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    }

    result = run(server.call_tool("retrieve_documentation", {"query": "query"}))
    assert result.structured_content == {  # type: ignore[union-attr]
        "result": [
            {
                "content": "First content",
                "location": "index.html#first",
                "title": "First",
                "breadcrumbs": ["Guide"],
                "distance": 0.0,
            },
            {
                "content": "Second content",
                "location": "index.html#second",
                "title": "Second",
                "breadcrumbs": ["Reference"],
                "distance": 9.0,
            },
        ]
    }


@pytest.mark.parametrize(
    "query, message",
    [("", "must not be empty"), ("x" * (MAX_QUERY_LENGTH + 1), "at most")],
)
def test_tool_translates_query_errors_to_safe_tool_errors(tmp_path: Path, query: str, message: str) -> None:
    server = create_server(
        RuntimeConfig(model="fake-model", artifact_directory=tmp_path),
        artifacts=make_artifacts(tmp_path),
        provider=FakeEmbeddingProvider(),
    )

    with pytest.raises(ToolError, match=message):
        run(server.call_tool("retrieve_documentation", {"query": query}))


class FailingEmbeddingProvider(FakeEmbeddingProvider):
    async def embed(self, inputs: Sequence[str]) -> NDArray[np.float32]:
        raise EmbeddingError("private provider response", safe_message="Embedding service unavailable.")


class FailingSearchIndex(FakeSearchIndex):
    def __init__(self) -> None:
        super().__init__(0)
        self.ntotal = 2

    def search(self, query: NDArray[np.float32], result_count: int) -> tuple[np.ndarray, np.ndarray]:
        raise RuntimeError("private index failure")


def test_tool_translates_embedding_and_retrieval_failures_safely(tmp_path: Path) -> None:
    embedding_server = create_server(
        RuntimeConfig(model="fake-model", artifact_directory=tmp_path),
        artifacts=make_artifacts(tmp_path),
        provider=FailingEmbeddingProvider(),
    )
    with pytest.raises(ToolError, match="Embedding service unavailable") as embedding_error:
        run(embedding_server.call_tool("retrieve_documentation", {"query": "query"}))
    assert "private provider response" not in str(embedding_error.value)

    retrieval_server = create_server(
        RuntimeConfig(model="fake-model", artifact_directory=tmp_path),
        artifacts=make_artifacts(tmp_path, FailingSearchIndex()),
        provider=FakeEmbeddingProvider(),
    )
    with pytest.raises(ToolError, match="index search failed") as retrieval_error:
        run(retrieval_server.call_tool("retrieve_documentation", {"query": "query"}))
    assert "private index failure" not in str(retrieval_error.value)


def test_tool_translates_invalid_structured_output_safely(tmp_path: Path) -> None:
    server = create_server(
        RuntimeConfig(model="fake-model", artifact_directory=tmp_path),
        artifacts=make_artifacts(tmp_path, title=""),
        provider=FakeEmbeddingProvider(),
    )

    with pytest.raises(ToolError, match="invalid snippet"):
        run(server.call_tool("retrieve_documentation", {"query": "query"}))


def test_server_startup_does_not_change_artifact_bytes_hashes_or_mtimes(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "index.html").write_text("raw html", encoding="utf-8")
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
    output = tmp_path / "db"
    run(build_index(source, output, FakeEmbeddingProvider(), model="fake-model"))
    paths = (output / "TA_Docu.faiss", output / "TA_Docu.documents.json", output / "TA_Docu.manifest.json")
    before = {
        path.name: (path.read_bytes(), sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns) for path in paths
    }

    create_server(
        RuntimeConfig(model="fake-model", artifact_directory=output),
        provider=FakeEmbeddingProvider(),
    )

    after = {
        path.name: (path.read_bytes(), sha256(path.read_bytes()).hexdigest(), path.stat().st_mtime_ns) for path in paths
    }
    assert after == before


def test_client_session_invokes_typed_tool_over_in_memory_transport(tmp_path: Path) -> None:
    provider = FakeEmbeddingProvider()
    server = create_server(
        RuntimeConfig(model="fake-model", artifact_directory=tmp_path),
        artifacts=make_artifacts(tmp_path),
        provider=provider,
    )

    async def exercise() -> tuple[object, bool]:
        async with (
            InMemoryTransport(server) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()
            result = await session.call_tool("retrieve_documentation", {"query": "query"})
            return result.structured_content, result.is_error

    structured_content, is_error = run(exercise())
    assert is_error is False
    assert structured_content == {
        "result": [
            {
                "content": "First content",
                "location": "index.html#first",
                "title": "First",
                "breadcrumbs": ["Guide"],
                "distance": 0.0,
            },
            {
                "content": "Second content",
                "location": "index.html#second",
                "title": "Second",
                "breadcrumbs": ["Reference"],
                "distance": 9.0,
            },
        ]
    }
    assert provider.closed


def test_client_session_returns_safe_protocol_error_for_embedding_failure(tmp_path: Path) -> None:
    server = create_server(
        RuntimeConfig(model="fake-model", artifact_directory=tmp_path),
        artifacts=make_artifacts(tmp_path),
        provider=FailingEmbeddingProvider(),
    )

    async def exercise() -> tuple[bool, str]:
        async with (
            InMemoryTransport(server) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as session,
        ):
            await session.initialize()
            result = await session.call_tool("retrieve_documentation", {"query": "query"})
            return result.is_error, str(result.content)

    is_error, content = run(exercise())
    assert is_error is True
    assert "Embedding service unavailable" in content
    assert "private provider response" not in content


def test_model_mismatch_is_rejected_before_server_start(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="does not match"):
        create_server(
            RuntimeConfig(model="other-model", artifact_directory=tmp_path),
            artifacts=make_artifacts(tmp_path),
            provider=FakeEmbeddingProvider(),
        )


def test_provider_is_closed_on_normal_and_exceptional_lifespan_exit(tmp_path: Path) -> None:
    async def exercise() -> tuple[bool, bool]:
        normal_provider = FakeEmbeddingProvider()
        normal_server = create_server(
            RuntimeConfig(model="fake-model", artifact_directory=tmp_path),
            artifacts=make_artifacts(tmp_path),
            provider=normal_provider,
        )
        lifespan = normal_server.settings.lifespan
        assert lifespan is not None
        async with lifespan(normal_server):
            pass

        exceptional_provider = FakeEmbeddingProvider()
        exceptional_server = create_server(
            RuntimeConfig(model="fake-model", artifact_directory=tmp_path),
            artifacts=make_artifacts(tmp_path),
            provider=exceptional_provider,
        )
        exceptional_lifespan = exceptional_server.settings.lifespan
        assert exceptional_lifespan is not None
        with pytest.raises(RuntimeError, match="shutdown"):
            async with exceptional_lifespan(exceptional_server):
                raise RuntimeError("shutdown")
        return normal_provider.closed, exceptional_provider.closed

    assert run(exercise()) == (True, True)


def test_provider_factory_is_created_once_and_serves_in_process_calls(tmp_path: Path) -> None:
    providers: list[FakeEmbeddingProvider] = []

    def factory(config: RuntimeConfig) -> FakeEmbeddingProvider:
        provider = FakeEmbeddingProvider()
        providers.append(provider)
        return provider

    server = create_server(
        RuntimeConfig(model="fake-model", artifact_directory=tmp_path),
        artifacts=make_artifacts(tmp_path),
        provider_factory=factory,
    )
    lifespan = server.settings.lifespan
    assert lifespan is not None

    async def exercise() -> object:
        async with lifespan(server):
            return await server.call_tool("retrieve_documentation", {"query": "query"})

    result = run(exercise())
    assert len(providers) == 1
    assert providers[0].closed
    assert result.structured_content is not None  # type: ignore[union-attr]
