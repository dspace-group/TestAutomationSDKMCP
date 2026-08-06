"""Read-only documentation retrieval and MCP server boundary."""

import asyncio
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from importlib.metadata import version
from math import isfinite
from pathlib import Path
from typing import Annotated, Protocol, cast, runtime_checkable

import numpy as np
from mcp.server import MCPServer
from mcp.server.mcpserver.context import Context
from mcp.types import ToolAnnotations
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .config import DEFAULT_ARTIFACT_DIRECTORY, RuntimeConfig
from .documents import EMBEDDING_DIMENSION
from .errors import ApplicationErrorCode, ConfigurationError, RetrievalError, TestAutomationSDKError, to_mcp_error
from .index import LoadedArtifacts, load_packaged_artifacts, load_verified_artifacts
from .ollama import EmbeddingProvider, OllamaEmbeddingProvider

MAX_QUERY_LENGTH = 4_000

SERVER_INSTRUCTIONS = (
    "Use this server when a user needs authoritative Test Automation SDK information or is developing, reviewing, "
    "or debugging Python tests that use dspace.testautomation. Call retrieve_documentation before making "
    "SDK-specific claims or choosing SDK APIs and patterns. Relevant topics include test environments and ports, "
    "variable access, capturing and triggers, scenario setup and execution, synchronization, and captured-data "
    "analysis. Do not call it solely for generic Python or pytest questions, test-domain logic that does not depend "
    "on the SDK, or unrelated dSPACE products. Base SDK-specific guidance on the returned snippets and retain their "
    "source locations."
)

TOOL_DESCRIPTION = (
    "Retrieve authoritative, sourced Test Automation SDK documentation for SDK questions and SDK-based Python test "
    "development. Use for exact APIs and patterns involving TestEnvironmentAccess or the ta fixture, variables, "
    "ports, Capturing or MultiPortCapturing, rasters and triggers, Scenario parameterization and lifecycle, "
    "synchronization, or capture result analysis. Do not use for generic Python or pytest questions or unrelated "
    "dSPACE products. Preconditions: verified packaged artifacts are loaded, the configured embedding model matches "
    "the index, and the configured Ollama-compatible endpoint is reachable and supports /api/embed. Success returns "
    "nearest-first documentation snippets with source locations and uncalibrated FAISS L2 distances. Public failures "
    "are invalid_query (permanent), embedding_request_timed_out (transient), embedding_service_unavailable "
    "(transient), embedding_service_rejected_request (permanent), embedding_service_failure (transient), "
    "embedding_response_invalid (permanent), retrieval_failed (permanent), tool_output_invalid (permanent), and "
    "server_runtime_unavailable (transient). Retry only transient failures with bounded backoff; change the input or "
    "configuration for permanent failures."
)


class DocumentationSnippet(BaseModel):
    """Strict structured output returned by the MCP retrieval tool."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    content: str
    location: str
    title: str
    breadcrumbs: list[str]
    distance: float = Field(description="L2 distance returned by the FAISS index.")

    @field_validator("content", "location", "title")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("snippet text fields must not be empty")
        return value

    @field_validator("distance")
    @classmethod
    def validate_distance(cls, value: float) -> float:
        if not isfinite(value) or value < 0:
            raise ValueError("distance must be a finite non-negative L2 distance")
        return value


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """Internal immutable representation of one nearest documentation result."""

    content: str
    location: str
    title: str
    breadcrumbs: tuple[str, ...]
    distance: float


@dataclass(frozen=True, slots=True)
class _ServerRuntime:
    retriever: "DocumentationRetriever"


@runtime_checkable
class _ClosableEmbeddingProvider(Protocol):
    async def aclose(self) -> None:
        """Close resources owned by the provider."""


EmbeddingProviderFactory = Callable[[RuntimeConfig], EmbeddingProvider]


def _validate_query(query: object) -> str:
    if not isinstance(query, str):
        raise RetrievalError("Documentation query must be a string.", code=ApplicationErrorCode.INVALID_QUERY)
    if not query.strip():
        raise RetrievalError("Documentation query must not be empty.", code=ApplicationErrorCode.INVALID_QUERY)
    if len(query) > MAX_QUERY_LENGTH:
        raise RetrievalError(
            f"Documentation query must be at most {MAX_QUERY_LENGTH} characters.",
            code=ApplicationErrorCode.INVALID_QUERY,
        )
    return query


def _validated_query_vector(vectors: object) -> NDArray[np.float32]:
    try:
        matrix = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
        finite = bool(np.isfinite(matrix).all())
    except (OverflowError, TypeError, ValueError) as error:
        raise RetrievalError(
            "Embedding provider returned an invalid query vector.",
            code=ApplicationErrorCode.EMBEDDING_RESPONSE_INVALID,
        ) from error
    if matrix.shape != (1, EMBEDDING_DIMENSION) or not finite:
        raise RetrievalError(
            "Embedding provider returned an invalid query vector.",
            code=ApplicationErrorCode.EMBEDDING_RESPONSE_INVALID,
        )
    return matrix


def _search_index(index: object, query_vector: NDArray[np.float32], result_count: int) -> tuple[object, object]:
    search = getattr(index, "search", None)
    if not callable(search):
        raise RetrievalError(
            "The documentation index cannot execute a search.", code=ApplicationErrorCode.RETRIEVAL_FAILED
        )
    try:
        result = search(query_vector, result_count)
    except (RuntimeError, TypeError, ValueError) as error:
        raise RetrievalError(
            "The documentation index search failed.", code=ApplicationErrorCode.RETRIEVAL_FAILED
        ) from error
    if not isinstance(result, tuple):
        raise RetrievalError(
            "The documentation index returned an invalid search result.", code=ApplicationErrorCode.RETRIEVAL_FAILED
        )
    result_items = cast(tuple[object, ...], result)
    if len(result_items) != 2:
        raise RetrievalError(
            "The documentation index returned an invalid search result.", code=ApplicationErrorCode.RETRIEVAL_FAILED
        )
    return result_items[0], result_items[1]


class DocumentationRetriever:
    """Embed queries and map validated FAISS rows to documentation records."""

    def __init__(self, artifacts: LoadedArtifacts, provider: EmbeddingProvider, result_count: int) -> None:
        if isinstance(result_count, bool) or result_count <= 0:
            raise ConfigurationError("result_count must be a positive integer.")
        self._artifacts = artifacts
        self._provider = provider
        self._result_count = result_count

    @classmethod
    def from_directory(
        cls,
        artifact_directory: Path,
        provider: EmbeddingProvider,
        result_count: int,
    ) -> "DocumentationRetriever":
        """Load and validate one artifact generation before serving queries."""

        artifacts = load_verified_artifacts(artifact_directory)
        return cls(artifacts, provider, result_count)

    async def retrieve(self, query: str) -> list[RetrievalResult]:
        """Return nearest-first results with FAISS L2 distances."""

        validated_query = _validate_query(query)
        documents = self._artifacts.documents.documents
        if not documents:
            return []

        index_count = getattr(self._artifacts.index, "ntotal", None)
        if isinstance(index_count, bool) or not isinstance(index_count, int) or index_count < 0:
            raise RetrievalError(
                "The documentation index has an invalid vector count.", code=ApplicationErrorCode.RETRIEVAL_FAILED
            )
        result_count = min(self._result_count, index_count, len(documents))
        if result_count == 0:
            return []

        vectors = await self._provider.embed([validated_query])
        query_vector = _validated_query_vector(vectors)

        try:
            raw_distances, raw_row_ids = await asyncio.to_thread(
                _search_index,
                self._artifacts.index,
                query_vector,
                result_count,
            )
        except TestAutomationSDKError:
            raise
        except (RuntimeError, TypeError, ValueError) as error:
            raise RetrievalError(
                "The documentation index search failed.", code=ApplicationErrorCode.RETRIEVAL_FAILED
            ) from error

        try:
            distances = np.asarray(raw_distances, dtype=np.float64)
            row_ids = np.asarray(raw_row_ids)
            finite_distances = bool(np.isfinite(distances).all())
        except (OverflowError, TypeError, ValueError) as error:
            raise RetrievalError(
                "The documentation index returned invalid search data.", code=ApplicationErrorCode.RETRIEVAL_FAILED
            ) from error
        if distances.shape != (1, result_count) or not finite_distances:
            raise RetrievalError(
                "The documentation index returned invalid distances.", code=ApplicationErrorCode.RETRIEVAL_FAILED
            )
        if row_ids.shape != (1, result_count) or not np.issubdtype(row_ids.dtype, np.integer):
            raise RetrievalError(
                "The documentation index returned invalid row IDs.", code=ApplicationErrorCode.RETRIEVAL_FAILED
            )

        for distance in distances[0]:
            if distance < 0:
                raise RetrievalError(
                    "The documentation index returned an invalid L2 distance.",
                    code=ApplicationErrorCode.RETRIEVAL_FAILED,
                )

        ordered_positions = sorted(range(result_count), key=lambda position: float(distances[0][position]))
        results: list[RetrievalResult] = []
        for position in ordered_positions:
            row_id = int(row_ids[0][position])
            if row_id < 0 or row_id >= len(documents):
                raise RetrievalError(
                    "The documentation index returned an invalid row ID.", code=ApplicationErrorCode.RETRIEVAL_FAILED
                )
            document = documents[row_id]
            results.append(
                RetrievalResult(
                    content=document.content,
                    location=document.location,
                    title=document.title,
                    breadcrumbs=document.breadcrumbs,
                    distance=float(distances[0][position]),
                )
            )
        return results


def _default_provider_factory(config: RuntimeConfig) -> EmbeddingProvider:
    return OllamaEmbeddingProvider(
        config.endpoint_url,
        config.model,
        api_key=config.api_key,
        connect_timeout=config.connect_timeout,
        request_timeout=config.request_timeout,
    )


async def _close_provider(provider: EmbeddingProvider) -> None:
    if isinstance(provider, _ClosableEmbeddingProvider):
        await provider.aclose()


def _snippet(result: RetrievalResult) -> DocumentationSnippet:
    return DocumentationSnippet(
        content=result.content,
        location=result.location,
        title=result.title,
        breadcrumbs=list(result.breadcrumbs),
        distance=result.distance,
    )


def create_server(
    config: RuntimeConfig | None = None,
    *,
    artifacts: LoadedArtifacts | None = None,
    provider: EmbeddingProvider | None = None,
    provider_factory: EmbeddingProviderFactory | None = None,
) -> MCPServer[_ServerRuntime]:
    """Create the single-tool MCP server with verified, immutable artifacts."""

    runtime_config = RuntimeConfig.from_environment() if config is None else config
    if provider is not None and provider_factory is not None:
        raise ConfigurationError("provider and provider_factory cannot both be configured.")
    if artifacts is not None:
        loaded_artifacts = artifacts
    elif runtime_config.artifact_directory == DEFAULT_ARTIFACT_DIRECTORY:
        loaded_artifacts = load_packaged_artifacts()
    else:
        loaded_artifacts = load_verified_artifacts(runtime_config.artifact_directory)
    if loaded_artifacts.manifest.embedding_model != runtime_config.model:
        raise ConfigurationError("Configured embedding model does not match the documentation index.")
    factory = _default_provider_factory if provider_factory is None else provider_factory
    direct_retriever = (
        None
        if provider is None
        else DocumentationRetriever(
            loaded_artifacts,
            provider,
            runtime_config.result_count,
        )
    )
    active_runtime: _ServerRuntime | None = None

    @asynccontextmanager
    async def lifespan(_: MCPServer[_ServerRuntime]) -> AsyncGenerator[_ServerRuntime, None]:
        nonlocal active_runtime
        owned_provider = provider if provider is not None else factory(runtime_config)
        active_runtime = _ServerRuntime(
            retriever=DocumentationRetriever(
                loaded_artifacts,
                owned_provider,
                runtime_config.result_count,
            )
        )
        try:
            yield active_runtime
        finally:
            active_runtime = None
            await _close_provider(owned_provider)

    server = MCPServer[_ServerRuntime](
        name="Test Automation SDK",
        description="Authoritative documentation retrieval for the Test Automation SDK and SDK-based test development.",
        instructions=SERVER_INSTRUCTIONS,
        version=version("test-automation-sdk-mcp"),
        lifespan=lifespan,
    )

    async def retrieve_documentation(
        query: Annotated[
            str,
            Field(
                strict=True,
                description=(
                    "One self-contained Test Automation SDK question or test-development task, including relevant "
                    "API names, SDK concepts, desired behavior, or error details."
                ),
            ),
        ],
        context: Context[_ServerRuntime, object],
    ) -> list[DocumentationSnippet]:
        """Retrieve relevant documentation for one non-empty query."""

        try:
            if direct_retriever is not None:
                retriever = direct_retriever
            else:
                try:
                    runtime_context = context.request_context.lifespan_context
                    retriever = runtime_context.retriever
                except (AttributeError, ValueError) as error:
                    if active_runtime is None:
                        project_error = RetrievalError(
                            "The documentation server runtime is unavailable.",
                            code=ApplicationErrorCode.SERVER_RUNTIME_UNAVAILABLE,
                        )
                        raise to_mcp_error(project_error) from error
                    retriever = active_runtime.retriever
            results = await retriever.retrieve(query)
            try:
                return [_snippet(result) for result in results]
            except ValidationError as error:
                project_error = RetrievalError(
                    "The documentation server returned an invalid snippet.",
                    code=ApplicationErrorCode.TOOL_OUTPUT_INVALID,
                )
                raise to_mcp_error(project_error) from error
        except TestAutomationSDKError as error:
            if error.code is None:
                raise
            raise to_mcp_error(error) from error

    server.add_tool(
        retrieve_documentation,
        name="retrieve_documentation",
        description=TOOL_DESCRIPTION,
        annotations=ToolAnnotations.model_validate(
            {
                "readOnlyHint": True,
                "destructiveHint": False,
                "idempotentHint": True,
                "openWorldHint": True,
            }
        ),
        structured_output=True,
    )

    return server


__all__ = [
    "MAX_QUERY_LENGTH",
    "SERVER_INSTRUCTIONS",
    "TOOL_DESCRIPTION",
    "DocumentationRetriever",
    "DocumentationSnippet",
    "EmbeddingProviderFactory",
    "RetrievalResult",
    "create_server",
]
