import asyncio
import json
from collections.abc import Awaitable, Callable
from typing import Any

import httpx
import numpy as np
import pytest
from pydantic import ValidationError

from test_automation_sdk_mcp.errors import ApplicationErrorCode, EmbeddingError
from test_automation_sdk_mcp.ollama import (
    EMBEDDING_DIMENSION,
    OllamaEmbeddingProvider,
    OllamaEmbeddingRequest,
    OllamaEmbeddingResponse,
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


def embedding_payload(inputs: list[str], model: str = "nomic-embed-text:v1.5") -> dict[str, object]:
    return {
        "model": model,
        "embeddings": [[float(index)] * EMBEDDING_DIMENSION for index in range(len(inputs))],
    }


def response_handler(payload: object, status_code: int = 200) -> ResponseHandler:
    async def handle(request: httpx.Request) -> httpx.Response:
        content = json.dumps(payload, allow_nan=True).encode("utf-8")
        return httpx.Response(
            status_code, content=content, headers={"content-type": "application/json"}, request=request
        )

    return handle


def test_request_url_payload_and_optional_bearer_header() -> None:
    transport = TrackingTransport(response_handler(embedding_payload(["one"])))
    provider = OllamaEmbeddingProvider(
        "https://embedding.example.test///",
        "nomic-embed-text:v1.5",
        api_key="secret-token",
        transport=transport,
    )

    result = run(provider.embed(["one"]))
    request = transport.requests[0]

    assert str(request.url) == "https://embedding.example.test/api/embed"
    assert json.loads(request.content) == {"model": "nomic-embed-text:v1.5", "input": ["one"]}
    assert request.headers["Authorization"] == "Bearer secret-token"
    assert result.shape == (1, EMBEDDING_DIMENSION)
    run(provider.aclose())

    no_auth_transport = TrackingTransport(response_handler(embedding_payload(["one"])))
    no_auth_provider = OllamaEmbeddingProvider(
        "http://embedding.example.test",
        "nomic-embed-text:v1.5",
        transport=no_auth_transport,
    )
    run(no_auth_provider.embed(["one"]))
    assert "Authorization" not in no_auth_transport.requests[0].headers
    run(no_auth_provider.aclose())


def test_multiple_inputs_are_sent_in_one_payload() -> None:
    transport = TrackingTransport(response_handler(embedding_payload(["one", "two"])))
    provider = OllamaEmbeddingProvider("http://embedding.example.test", "nomic-embed-text:v1.5", transport=transport)

    run(provider.embed(["one", "two"]))

    assert json.loads(transport.requests[0].content)["input"] == ["one", "two"]
    run(provider.aclose())


@pytest.mark.parametrize("model", ["", "  ", "model\nname"])
def test_provider_rejects_invalid_model_names(model: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        OllamaEmbeddingProvider("http://embedding.example.test", model)


def test_request_rejects_unknown_fields_and_response_ignores_telemetry() -> None:
    with pytest.raises(ValidationError):
        OllamaEmbeddingRequest.model_validate({"model": "nomic-embed-text:v1.5", "input": ("one",), "unexpected": True})

    response = OllamaEmbeddingResponse.model_validate(
        {
            "model": "nomic-embed-text:v1.5",
            "embeddings": [[0.0] * EMBEDDING_DIMENSION],
            "total_duration": 123,
            "load_duration": 45,
            "prompt_eval_count": 10,
        }
    )

    assert response.model == "nomic-embed-text:v1.5"
    assert response.model_dump() == {
        "model": "nomic-embed-text:v1.5",
        "embeddings": [[0.0] * EMBEDDING_DIMENSION],
    }


def test_batches_preserve_input_order_and_return_contiguous_float32_matrix() -> None:
    vector_ids = {"one": 11.0, "two": 22.0, "three": 33.0, "four": 44.0, "five": 55.0}

    async def handle(request: httpx.Request) -> httpx.Response:
        inputs = json.loads(request.content)["input"]
        payload = {
            "embeddings": [[vector_ids[input_value]] * EMBEDDING_DIMENSION for input_value in inputs],
        }
        return httpx.Response(200, json=payload, request=request)

    transport = TrackingTransport(handle)
    provider = OllamaEmbeddingProvider(
        "http://embedding.example.test",
        "nomic-embed-text:v1.5",
        batch_size=2,
        transport=transport,
    )

    result = run(provider.embed(["one", "two", "three", "four", "five"]))

    assert [json.loads(request.content)["input"] for request in transport.requests] == [
        ["one", "two"],
        ["three", "four"],
        ["five"],
    ]
    assert result.shape == (5, EMBEDDING_DIMENSION)
    assert result.dtype == np.dtype(np.float32)
    assert result.flags.c_contiguous
    assert result[:, 0].tolist() == [11.0, 22.0, 33.0, 44.0, 55.0]
    run(provider.aclose())


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"unexpected": True}, "invalid response"),
        ({"model": "other", "embeddings": [[0.0] * EMBEDDING_DIMENSION]}, "unexpected model"),
        ({"embeddings": []}, "unexpected number of vectors"),
        ({"embeddings": [[0.0] * EMBEDDING_DIMENSION, [0.0] * EMBEDDING_DIMENSION]}, "unexpected number of vectors"),
        ({"embeddings": [["not-a-number"] * EMBEDDING_DIMENSION]}, "invalid response"),
        ({"embeddings": [[0.0] * (EMBEDDING_DIMENSION - 1)]}, "768 dimensions"),
        ({"embeddings": [[float("nan")] * EMBEDDING_DIMENSION]}, "invalid vector values"),
        ({"embeddings": [[10**400] * EMBEDDING_DIMENSION]}, "invalid vector values"),
    ],
)
def test_response_validation_errors_are_safe(payload: object, message: str) -> None:
    transport = TrackingTransport(response_handler(payload))
    provider = OllamaEmbeddingProvider("http://embedding.example.test", "nomic-embed-text:v1.5", transport=transport)

    with pytest.raises(EmbeddingError, match=message) as raised:
        run(provider.embed(["one"]))

    assert raised.value.code is ApplicationErrorCode.EMBEDDING_RESPONSE_INVALID
    assert "not-a-number" not in str(transport.requests[0])
    run(provider.aclose())


@pytest.mark.parametrize(
    ("status_code", "expected_code"),
    [
        (408, ApplicationErrorCode.EMBEDDING_SERVICE_FAILURE),
        (429, ApplicationErrorCode.EMBEDDING_SERVICE_FAILURE),
        (503, ApplicationErrorCode.EMBEDDING_SERVICE_FAILURE),
        (400, ApplicationErrorCode.EMBEDDING_SERVICE_REJECTED_REQUEST),
        (404, ApplicationErrorCode.EMBEDDING_SERVICE_REJECTED_REQUEST),
    ],
)
def test_http_status_failures_are_classified(status_code: int, expected_code: ApplicationErrorCode) -> None:
    transport = TrackingTransport(response_handler({"private": "body"}, status_code=status_code))
    provider = OllamaEmbeddingProvider("http://embedding.example.test", "nomic-embed-text:v1.5", transport=transport)

    with pytest.raises(EmbeddingError, match=f"HTTP status {status_code}") as raised:
        run(provider.embed(["one"]))

    assert raised.value.code is expected_code
    assert "private" not in str(raised.value)
    run(provider.aclose())


def test_malformed_json_timeout_and_network_failures_are_classified() -> None:
    async def malformed(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"{not-json", request=request)

    async def timeout(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=request)

    async def connection_error(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection failed", request=request)

    for handler, message, expected_code in (
        (malformed, "invalid JSON", ApplicationErrorCode.EMBEDDING_RESPONSE_INVALID),
        (timeout, "timed out", ApplicationErrorCode.EMBEDDING_REQUEST_TIMED_OUT),
        (connection_error, "unavailable", ApplicationErrorCode.EMBEDDING_SERVICE_UNAVAILABLE),
    ):
        transport = TrackingTransport(handler)
        provider = OllamaEmbeddingProvider(
            "http://embedding.example.test", "nomic-embed-text:v1.5", transport=transport
        )

        with pytest.raises(EmbeddingError, match=message) as raised:
            run(provider.embed(["one"]))

        assert raised.value.code is expected_code
        assert raised.value.__cause__ is not None
        run(provider.aclose())


def test_client_is_closed_after_success_and_failure() -> None:
    success_transport = TrackingTransport(response_handler({"embeddings": [[0.0] * EMBEDDING_DIMENSION]}))
    success_provider = OllamaEmbeddingProvider(
        "http://embedding.example.test", "nomic-embed-text:v1.5", transport=success_transport
    )
    run(success_provider.embed(["one"]))
    run(success_provider.aclose())
    run(success_provider.aclose())
    assert success_transport.closed

    failure_transport = TrackingTransport(response_handler({"embeddings": []}))

    async def exercise_failure() -> None:
        async with OllamaEmbeddingProvider(
            "http://embedding.example.test", "nomic-embed-text:v1.5", transport=failure_transport
        ) as failure_provider:
            with pytest.raises(EmbeddingError):
                await failure_provider.embed(["one"])

    run(exercise_failure())
    assert failure_transport.closed
