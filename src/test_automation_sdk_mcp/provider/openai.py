"""OpenAI-compatible embeddings transport."""

from collections.abc import Sequence
from math import isfinite
from typing import Literal, Self
from urllib.parse import urlsplit

import httpx
import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, ValidationError

from ..config import EmbeddingProviderKind
from ..errors import ApplicationErrorCode, EmbeddingError
from .embeddings import (
    DEFAULT_BATCH_SIZE,
    EMBEDDING_DIMENSION,
    EmbeddingProfile,
    validate_batch_size,
    validate_inputs,
    validate_model,
    validate_timeout,
    vectors_to_matrix,
)


class _RequestBoundaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class _ResponseBoundaryModel(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)


class OpenAIEmbeddingRequest(_RequestBoundaryModel):
    """Project-owned request payload for an OpenAI-compatible endpoint."""

    model: str | None = None
    input: tuple[str, ...]
    encoding_format: Literal["float"] = "float"


class OpenAIEmbeddingData(_ResponseBoundaryModel):
    """One externally-owned OpenAI-compatible embedding response item."""

    object: Literal["embedding"]
    index: int
    embedding: list[float | int]


class OpenAIEmbeddingResponse(_ResponseBoundaryModel):
    """Externally-owned response boundary; additive fields are ignored."""

    data: list[OpenAIEmbeddingData]
    model: str | None = None


def _normalize_endpoint(endpoint_url: str) -> str:
    normalized = endpoint_url.strip().rstrip("/")
    if not normalized:
        raise ValueError("endpoint_url must not be empty")
    parsed = urlsplit(normalized)
    if (
        parsed.scheme not in {"http", "https"}
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("endpoint_url must be a valid HTTP or HTTPS URL")
    return normalized


def _invalid_response(message: str) -> EmbeddingError:
    return EmbeddingError(message, code=ApplicationErrorCode.EMBEDDING_RESPONSE_INVALID)


class OpenAIEmbeddingProvider:
    """Batch requests to an OpenAI-compatible embeddings endpoint."""

    def __init__(
        self,
        endpoint_url: str,
        model: str | None = None,
        *,
        api_key: str | None = None,
        connect_timeout: float = 5.0,
        request_timeout: float = 30.0,
        batch_size: int = DEFAULT_BATCH_SIZE,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._endpoint_url = _normalize_endpoint(endpoint_url)
        self._model = None if model is None else validate_model(model)
        self._profile = EmbeddingProfile(EmbeddingProviderKind.OPENAI, self._model)
        self._batch_size = validate_batch_size(batch_size)
        self._closed = False
        request_timeout_value = validate_timeout(request_timeout, "request_timeout")
        connect_timeout_value = validate_timeout(connect_timeout, "connect_timeout")
        timeout = httpx.Timeout(request_timeout_value, connect=connect_timeout_value)
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._client = httpx.AsyncClient(headers=headers, timeout=timeout, transport=transport)

    @property
    def profile(self) -> EmbeddingProfile:
        return self._profile

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
        """Embed a non-empty sequence in bounded requests and restore response order."""

        if self._closed:
            raise EmbeddingError(
                "Embedding provider is closed.", code=ApplicationErrorCode.EMBEDDING_SERVICE_UNAVAILABLE
            )
        input_values = validate_inputs(inputs)

        vectors: list[list[float | int]] = []
        for start in range(0, len(input_values), self._batch_size):
            batch = input_values[start : start + self._batch_size]
            vectors.extend(await self._embed_batch(batch))
        return vectors_to_matrix(vectors, len(input_values))

    async def _embed_batch(self, inputs: tuple[str, ...]) -> list[list[float | int]]:
        request = OpenAIEmbeddingRequest(model=self._model, input=inputs)
        payload = request.model_dump(mode="json", exclude_none=True)
        try:
            response = await self._client.post(self._endpoint_url, json=payload)
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
            response_payload = response.json()
        except (TypeError, ValueError, UnicodeError) as error:
            raise _invalid_response("Embedding service returned invalid JSON.") from error
        try:
            parsed = OpenAIEmbeddingResponse.model_validate(response_payload)
        except ValidationError as error:
            raise _invalid_response("Embedding service returned an invalid response.") from error

        if self._model is not None and parsed.model is not None and parsed.model != self._model:
            raise _invalid_response("Embedding service returned an unexpected model.")
        if len(parsed.data) != len(inputs):
            raise _invalid_response("Embedding service returned an unexpected number of vectors.")

        by_index: dict[int, list[float | int]] = {}
        for item in parsed.data:
            if item.index < 0 or item.index >= len(inputs) or item.index in by_index:
                raise _invalid_response("Embedding service returned invalid vector indexes.")
            if len(item.embedding) != EMBEDDING_DIMENSION:
                raise _invalid_response(f"Embedding vectors must have {EMBEDDING_DIMENSION} dimensions.")
            vector: list[float | int] = []
            for value in item.embedding:
                if isinstance(value, bool):
                    raise _invalid_response("Embedding service returned invalid vector values.")
                try:
                    numeric_value = float(value)
                except (OverflowError, TypeError, ValueError) as error:
                    raise _invalid_response("Embedding service returned invalid vector values.") from error
                if not isfinite(numeric_value):
                    raise _invalid_response("Embedding service returned invalid vector values.")
                vector.append(value)
            by_index[item.index] = vector

        if set(by_index) != set(range(len(inputs))):
            raise _invalid_response("Embedding service returned invalid vector indexes.")
        return [by_index[index] for index in range(len(inputs))]


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "EMBEDDING_DIMENSION",
    "OpenAIEmbeddingData",
    "OpenAIEmbeddingProvider",
    "OpenAIEmbeddingRequest",
    "OpenAIEmbeddingResponse",
]
