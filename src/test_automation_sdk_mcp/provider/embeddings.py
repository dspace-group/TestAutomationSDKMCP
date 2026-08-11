"""Provider-neutral embedding validation and contracts."""

from collections.abc import Sequence
from dataclasses import dataclass
from math import isfinite
from typing import Protocol, cast

import numpy as np
from numpy.typing import NDArray

from ..config import EmbeddingProviderKind
from ..errors import ApplicationErrorCode, EmbeddingError

EMBEDDING_DIMENSION = 768
DEFAULT_BATCH_SIZE = 32


@dataclass(frozen=True, slots=True)
class EmbeddingProvenance:
    """Immutable provider/model provenance recorded in generated artifacts."""

    provider: EmbeddingProviderKind
    model: str | None

    def __post_init__(self) -> None:
        provider = EmbeddingProviderKind(self.provider)
        object.__setattr__(self, "provider", provider)
        if provider is EmbeddingProviderKind.OLLAMA and self.model is None:
            raise ValueError("Ollama embedding provenance requires a model.")
        if self.model is not None:
            object.__setattr__(self, "model", validate_model(self.model))


class EmbeddingProvider(Protocol):
    """Async batch embedding contract used by index builders and retrieval."""

    async def embed(self, inputs: Sequence[str]) -> NDArray[np.float32]:
        """Embed inputs in their original order."""

        ...

    async def aclose(self) -> None:
        """Release resources owned by the provider; resource-free providers may no-op."""

        ...


def validate_model(value: object) -> str:
    if not isinstance(value, str):
        raise TypeError("model must be a non-empty model name")
    model = value.strip()
    if not model or any(ord(character) < 32 for character in model):
        raise ValueError("model must be a non-empty model name")
    return model


def validate_timeout(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a positive finite number")
    timeout = float(value)
    if timeout <= 0 or not isfinite(timeout):
        raise ValueError(f"{name} must be a positive finite number")
    return timeout


def validate_batch_size(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("batch_size must be a positive integer")
    if value <= 0:
        raise ValueError("batch_size must be a positive integer")
    return value


def validate_inputs(value: object) -> tuple[str, ...]:
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


def vectors_to_matrix(vectors: Sequence[Sequence[float | int]], input_count: int) -> NDArray[np.float32]:
    """Convert provider vectors to one contiguous float32 matrix."""

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


__all__ = [
    "DEFAULT_BATCH_SIZE",
    "EMBEDDING_DIMENSION",
    "EmbeddingProvenance",
    "EmbeddingProvider",
    "validate_batch_size",
    "validate_inputs",
    "validate_model",
    "validate_timeout",
    "vectors_to_matrix",
]
