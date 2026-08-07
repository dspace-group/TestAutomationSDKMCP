import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import numpy as np
import pytest
from pydantic import ValidationError

from test_automation_sdk_mcp.errors import ApplicationErrorCode, EmbeddingError
from test_automation_sdk_mcp.provider.openai import (
    EMBEDDING_DIMENSION,
    OpenAIEmbeddingData,
    OpenAIEmbeddingProvider,
    OpenAIEmbeddingRequest,
    OpenAIEmbeddingResponse,
)

ResponseHandler = Callable[[httpx.Request], Awaitable[httpx.Response]]


class TrackingTransport(httpx.AsyncBaseTransport):
    def __init__(self, handler: ResponseHandler) -> None:
        self.handler = handler
        self.requests: list[httpx.Request] = []
        self.closed = False

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return await self.handler(request)

    async def aclose(self) -> None:
        self.closed = True


def run(coroutine: Awaitable[Any]) -> Any:
    return asyncio.run(coroutine)


def response_payload(inputs: list[str], model: str | None = "endpoint-model") -> dict[str, object]:
    data = [
        {"object": "embedding", "index": index, "embedding": [float(index)] * EMBEDDING_DIMENSION}
        for index, _ in enumerate(inputs)
    ]
    payload: dict[str, object] = {"data": data, "usage": {"prompt_tokens": len(inputs)}}
    if model is not None:
        payload["model"] = model
    return payload


def response_handler(payload: object, status_code: int = 200) -> ResponseHandler:
    async def handle(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code, content=json.dumps(payload, allow_nan=True), request=request)

    return handle


def test_request_contract_auth_and_response_order() -> None:
    async def handle(request: httpx.Request) -> httpx.Response:
        request_payload = json.loads(request.content)
        assert request_payload == {
            "model": "endpoint-model",
            "input": ["one", "two"],
            "encoding_format": "float",
        }
        return httpx.Response(
            200,
            json={
                "model": "endpoint-model",
                "data": [
                    {"object": "embedding", "index": 1, "embedding": [2.0] * EMBEDDING_DIMENSION},
                    {"object": "embedding", "index": 0, "embedding": [1.0] * EMBEDDING_DIMENSION},
                ],
                "usage": {"total_tokens": 2},
            },
            request=request,
        )

    transport = TrackingTransport(handle)
    provider = OpenAIEmbeddingProvider(
        "https://embedding.example.test/v1/embeddings/",
        "endpoint-model",
        api_key="secret-token",
        transport=transport,
    )

    result = run(provider.embed(["one", "two"]))

    assert str(transport.requests[0].url) == "https://embedding.example.test/v1/embeddings"
    assert transport.requests[0].headers["Authorization"] == "Bearer secret-token"
    assert result.shape == (2, EMBEDDING_DIMENSION)
    assert result.dtype == np.dtype(np.float32)
    assert result.flags.c_contiguous
    assert result[:, 0].tolist() == [1.0, 2.0]
    run(provider.aclose())


def test_model_is_omitted_without_binding_and_auth_is_optional() -> None:
    transport = TrackingTransport(response_handler(response_payload(["one"], model="active-model")))
    provider = OpenAIEmbeddingProvider("http://embedding.example.test/v1/embeddings", transport=transport)

    run(provider.embed(["one"]))

    assert json.loads(transport.requests[0].content) == {
        "input": ["one"],
        "encoding_format": "float",
    }
    assert "Authorization" not in transport.requests[0].headers
    run(provider.aclose())


def test_batches_preserve_input_order() -> None:
    async def handle(request: httpx.Request) -> httpx.Response:
        inputs = json.loads(request.content)["input"]
        return httpx.Response(200, json=response_payload(inputs), request=request)

    transport = TrackingTransport(handle)
    provider = OpenAIEmbeddingProvider(
        "http://embedding.example.test/v1/embeddings", "endpoint-model", batch_size=2, transport=transport
    )

    result = run(provider.embed(["one", "two", "three"]))

    assert [json.loads(request.content)["input"] for request in transport.requests] == [
        ["one", "two"],
        ["three"],
    ]
    assert result.shape == (3, EMBEDDING_DIMENSION)
    assert result[:, 0].tolist() == [0.0, 1.0, 0.0]
    run(provider.aclose())


@pytest.mark.parametrize(
    "indexes",
    [[0, 0], [-1, 1], [0, 2], [True, 1], [0.5, 1]],
)
def test_invalid_indexes_are_rejected(indexes: list[object]) -> None:
    payload = {
        "model": "endpoint-model",
        "data": [
            {"object": "embedding", "index": index, "embedding": [0.0] * EMBEDDING_DIMENSION} for index in indexes
        ],
    }
    transport = TrackingTransport(response_handler(payload))
    provider = OpenAIEmbeddingProvider(
        "http://embedding.example.test/v1/embeddings", "endpoint-model", transport=transport
    )

    with pytest.raises(EmbeddingError) as raised:
        run(provider.embed(["one", "two"]))

    assert raised.value.code is ApplicationErrorCode.EMBEDDING_RESPONSE_INVALID
    run(provider.aclose())


@pytest.mark.parametrize(
    "payload",
    [
        {"data": []},
        {"model": "other-model", "data": [{"object": "embedding", "index": 0, "embedding": [0.0] * 768}]},
        {"model": "endpoint-model", "data": [{"object": "embedding", "index": 0, "embedding": [0.0] * 767}]},
        {"model": "endpoint-model", "data": [{"object": "embedding", "index": 0, "embedding": [float("nan")] * 768}]},
        {"data": [{"object": "embedding", "index": 0, "embedding": ["bad"] * 768}]},
        {"data": [{"object": "embedding", "index": 0, "embedding": [10**400] * 768}]},
    ],
)
def test_response_validation_is_safe(payload: object) -> None:
    transport = TrackingTransport(response_handler(payload))
    provider = OpenAIEmbeddingProvider(
        "http://embedding.example.test/v1/embeddings", "endpoint-model", transport=transport
    )

    with pytest.raises(EmbeddingError) as raised:
        run(provider.embed(["one"]))

    assert raised.value.code is ApplicationErrorCode.EMBEDDING_RESPONSE_INVALID
    assert "bad" not in str(raised.value)
    run(provider.aclose())


def test_wire_models_have_strict_request_and_additive_response_boundaries() -> None:
    with pytest.raises(ValidationError):
        OpenAIEmbeddingRequest.model_validate({"input": ("one",), "encoding_format": "float", "unexpected": True})
    response = OpenAIEmbeddingResponse.model_validate(
        {
            "model": "endpoint-model",
            "data": [{"object": "embedding", "index": 0, "embedding": [0.0]}],
            "usage": {"total_tokens": 1},
        }
    )
    assert response.model == "endpoint-model"
    assert OpenAIEmbeddingData.model_fields["index"].annotation is int
    assert "usage" not in response.model_dump()


def test_close_is_idempotent_and_use_after_close_is_safe() -> None:
    transport = TrackingTransport(response_handler(response_payload(["one"])))
    provider = OpenAIEmbeddingProvider("http://embedding.example.test/v1/embeddings", transport=transport)

    run(provider.aclose())
    run(provider.aclose())
    with pytest.raises(EmbeddingError, match="closed") as raised:
        run(provider.embed(["one"]))

    assert raised.value.code is ApplicationErrorCode.EMBEDDING_SERVICE_UNAVAILABLE
    assert transport.closed
