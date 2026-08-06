"""Async embeddings through an Ollama-compatible HTTP endpoint."""

from collections.abc import Sequence
from math import isfinite
from typing import Protocol, Self, cast

import httpx
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, ValidationError

from .errors import ApplicationErrorCode, EmbeddingError

EMBEDDING_DIMENSION = 768
DEFAULT_BATCH_SIZE = 32


class EmbeddingProvider(Protocol):
    """Async batch embedding contract used by index builders and retrieval."""

    async def embed(self, inputs: Sequence[str]) -> NDArray[np.float32]:
        """Embed inputs in their original order."""

        ...


class _RequestBoundaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class _ResponseBoundaryModel(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)


class OllamaEmbeddingRequest(_RequestBoundaryModel):
    """Validated request payload for Ollama's embedding endpoint."""

    model: str
    input: tuple[str, ...]


class OllamaEmbeddingResponse(_ResponseBoundaryModel):
    """Validated response payload returned by Ollama's embedding endpoint."""

    model: str | None = None
    embeddings: list[list[float | int]]


def _validate_model(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("model must be a non-empty model name")
    model = value.strip()
    if not model or any(ord(character) < 32 for character in model):
        raise ValueError("model must be a non-empty model name")
    return model


def _normalize_endpoint(endpoint_url: str) -> str:
    normalized = endpoint_url.strip().rstrip("/")
    if not normalized:
        raise ValueError("endpoint_url must not be empty")
    return f"{normalized}/api/embed"


def _validate_timeout(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive finite number")
    timeout = float(value)
    if timeout <= 0 or not isfinite(timeout):
        raise ValueError(f"{name} must be a positive finite number")
    return timeout


def _validate_batch_size(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("batch_size must be a positive integer")
    if value <= 0:
        raise ValueError("batch_size must be a positive integer")
    return value


def _validate_inputs(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError("inputs must be a sequence of strings")
    validated: list[str] = []
    items = cast(Sequence[object], value)
    for item in items:
        if not isinstance(item, str):
            raise TypeError("inputs must be a sequence of strings")
        validated.append(item)
    if not validated:
        raise ValueError("inputs must not be empty")
    return tuple(validated)


def _vectors_to_matrix(vectors: Sequence[Sequence[float]], input_count: int) -> NDArray[np.float32]:
    try:
        matrix = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
    except (OverflowError, TypeError, ValueError) as error:
        raise EmbeddingError(
            "Embedding service returned invalid vector values.",
            code=ApplicationErrorCode.EMBEDDING_RESPONSE_INVALID,
        ) from error
    if matrix.shape != (input_count, EMBEDDING_DIMENSION) or not bool(np.isfinite(matrix).all()):
        raise EmbeddingError(
            "Embedding service returned invalid vector values.",
            code=ApplicationErrorCode.EMBEDDING_RESPONSE_INVALID,
        )
    return matrix


class OllamaEmbeddingProvider:
    """Own an HTTP client and validate all vectors returned by Ollama."""

    def __init__(
        self,
        endpoint_url: str,
        model: str,
        *,
        api_key: str | None = None,
        connect_timeout: float = 5.0,
        request_timeout: float = 30.0,
        batch_size: int = DEFAULT_BATCH_SIZE,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._endpoint_url = _normalize_endpoint(endpoint_url)
        self._model = _validate_model(model)
        self._batch_size = _validate_batch_size(batch_size)
        self._closed = False
        request_timeout_value = _validate_timeout(request_timeout, "request_timeout")
        connect_timeout_value = _validate_timeout(connect_timeout, "connect_timeout")
        timeout = httpx.Timeout(request_timeout_value, connect=connect_timeout_value)
        headers: dict[str, str] = {}
        if api_key is not None and api_key != "":
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(headers=headers, timeout=timeout, transport=transport)

    async def __aenter__(self) -> Self:
        if self._closed:
            raise EmbeddingError(
                "Embedding provider is closed.", code=ApplicationErrorCode.EMBEDDING_SERVICE_UNAVAILABLE
            )
        return self

    async def __aexit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the owned HTTP client; repeated cleanup is harmless."""

        if not self._closed:
            self._closed = True
            await self._client.aclose()

    async def embed(self, inputs: Sequence[str]) -> NDArray[np.float32]:
        """Embed a non-empty sequence in bounded requests and preserve its order."""

        if self._closed:
            raise EmbeddingError(
                "Embedding provider is closed.", code=ApplicationErrorCode.EMBEDDING_SERVICE_UNAVAILABLE
            )
        if isinstance(inputs, (str, bytes)):
            raise TypeError("inputs must be a sequence of strings")
        input_values = _validate_inputs(inputs)

        vectors: list[list[float]] = []
        for start in range(0, len(input_values), self._batch_size):
            batch = input_values[start : start + self._batch_size]
            vectors.extend(await self._embed_batch(batch))
        return _vectors_to_matrix(vectors, len(input_values))

    async def _embed_batch(self, inputs: tuple[str, ...]) -> list[list[float]]:
        request = OllamaEmbeddingRequest(model=self._model, input=inputs)
        try:
            response = await self._client.post(self._endpoint_url, json=request.model_dump(mode="json"))
            response.raise_for_status()
        except httpx.TimeoutException as error:
            raise EmbeddingError(
                "Embedding request timed out.", code=ApplicationErrorCode.EMBEDDING_REQUEST_TIMED_OUT
            ) from error
        except httpx.HTTPStatusError as error:
            status_code = error.response.status_code
            code = (
                ApplicationErrorCode.EMBEDDING_SERVICE_FAILURE
                if status_code in {408, 429} or status_code >= 500
                else ApplicationErrorCode.EMBEDDING_SERVICE_REJECTED_REQUEST
            )
            raise EmbeddingError(f"Embedding service returned HTTP status {status_code}.", code=code) from error
        except httpx.RequestError as error:
            raise EmbeddingError(
                "Embedding service is unavailable.", code=ApplicationErrorCode.EMBEDDING_SERVICE_UNAVAILABLE
            ) from error

        try:
            payload = response.json()
        except (TypeError, ValueError, UnicodeError) as error:
            raise EmbeddingError(
                "Embedding service returned invalid JSON.", code=ApplicationErrorCode.EMBEDDING_RESPONSE_INVALID
            ) from error
        try:
            parsed = OllamaEmbeddingResponse.model_validate(payload)
        except ValidationError as error:
            raise EmbeddingError(
                "Embedding service returned an invalid response.", code=ApplicationErrorCode.EMBEDDING_RESPONSE_INVALID
            ) from error

        if parsed.model is not None and parsed.model != self._model:
            raise EmbeddingError(
                "Embedding service returned an unexpected model.", code=ApplicationErrorCode.EMBEDDING_RESPONSE_INVALID
            )
        if len(parsed.embeddings) != len(inputs):
            raise EmbeddingError(
                "Embedding service returned an unexpected number of vectors.",
                code=ApplicationErrorCode.EMBEDDING_RESPONSE_INVALID,
            )

        vectors: list[list[float]] = []
        for vector in parsed.embeddings:
            if len(vector) != EMBEDDING_DIMENSION:
                raise EmbeddingError(
                    f"Embedding vectors must have {EMBEDDING_DIMENSION} dimensions.",
                    code=ApplicationErrorCode.EMBEDDING_RESPONSE_INVALID,
                )
            try:
                converted = [float(value) for value in vector]
            except (OverflowError, TypeError, ValueError) as error:
                raise EmbeddingError(
                    "Embedding service returned invalid vector values.",
                    code=ApplicationErrorCode.EMBEDDING_RESPONSE_INVALID,
                ) from error
            if not all(isfinite(value) for value in converted):
                raise EmbeddingError(
                    "Embedding service returned invalid vector values.",
                    code=ApplicationErrorCode.EMBEDDING_RESPONSE_INVALID,
                )
            vectors.append(converted)
        return vectors


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "EMBEDDING_DIMENSION",
    "EmbeddingProvider",
    "OllamaEmbeddingProvider",
    "OllamaEmbeddingRequest",
    "OllamaEmbeddingResponse",
]
