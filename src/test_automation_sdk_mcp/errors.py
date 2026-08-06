"""Project-specific exceptions and public MCP error contracts."""

from enum import Enum
from typing import Literal, Self

from mcp.shared.exceptions import MCPError
from pydantic import BaseModel, ConfigDict, model_validator

TOOL_EXECUTION_ERROR = -32000


class ApplicationErrorCode(str, Enum):
    """Stable application error codes exposed by MCP tool calls."""

    INVALID_QUERY = "invalid_query"
    EMBEDDING_REQUEST_TIMED_OUT = "embedding_request_timed_out"
    EMBEDDING_SERVICE_UNAVAILABLE = "embedding_service_unavailable"
    EMBEDDING_SERVICE_REJECTED_REQUEST = "embedding_service_rejected_request"
    EMBEDDING_SERVICE_FAILURE = "embedding_service_failure"
    EMBEDDING_RESPONSE_INVALID = "embedding_response_invalid"
    RETRIEVAL_FAILED = "retrieval_failed"
    TOOL_OUTPUT_INVALID = "tool_output_invalid"
    SERVER_RUNTIME_UNAVAILABLE = "server_runtime_unavailable"


class ErrorClassification(str, Enum):
    """Retry classification for public application errors."""

    TRANSIENT = "transient"
    PERMANENT = "permanent"


ERROR_CLASSIFICATIONS: dict[ApplicationErrorCode, ErrorClassification] = {
    ApplicationErrorCode.INVALID_QUERY: ErrorClassification.PERMANENT,
    ApplicationErrorCode.EMBEDDING_REQUEST_TIMED_OUT: ErrorClassification.TRANSIENT,
    ApplicationErrorCode.EMBEDDING_SERVICE_UNAVAILABLE: ErrorClassification.TRANSIENT,
    ApplicationErrorCode.EMBEDDING_SERVICE_REJECTED_REQUEST: ErrorClassification.PERMANENT,
    ApplicationErrorCode.EMBEDDING_SERVICE_FAILURE: ErrorClassification.TRANSIENT,
    ApplicationErrorCode.EMBEDDING_RESPONSE_INVALID: ErrorClassification.PERMANENT,
    ApplicationErrorCode.RETRIEVAL_FAILED: ErrorClassification.PERMANENT,
    ApplicationErrorCode.TOOL_OUTPUT_INVALID: ErrorClassification.PERMANENT,
    ApplicationErrorCode.SERVER_RUNTIME_UNAVAILABLE: ErrorClassification.TRANSIENT,
}


class PublicErrorEnvelope(BaseModel):
    """Strict versioned data attached to native MCP errors."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1] = 1
    code: ApplicationErrorCode
    classification: ErrorClassification
    retryable: bool

    @model_validator(mode="after")
    def validate_classification(self) -> Self:
        expected = ERROR_CLASSIFICATIONS[self.code]
        if self.classification is not expected or self.retryable is not (expected is ErrorClassification.TRANSIENT):
            raise ValueError("classification and retryable must match the application error code")
        return self

    @classmethod
    def for_code(cls, code: ApplicationErrorCode) -> "PublicErrorEnvelope":
        classification = ERROR_CLASSIFICATIONS[code]
        return cls(
            code=code,
            classification=classification,
            retryable=classification is ErrorClassification.TRANSIENT,
        )


class TestAutomationSDKError(Exception):
    """Base class for errors that may be translated at a public boundary."""

    __test__ = False

    def __init__(
        self,
        message: str,
        *,
        safe_message: str | None = None,
        code: ApplicationErrorCode | None = None,
    ) -> None:
        self.safe_message = safe_message if safe_message is not None else message
        self.code = code
        super().__init__(self.safe_message)


def to_mcp_error(error: TestAutomationSDKError) -> MCPError:
    """Convert one explicitly coded project exception to a native MCP error."""

    if error.code is None:
        raise ValueError("Project exception does not have a public application error code.")
    envelope = PublicErrorEnvelope.for_code(error.code)
    return MCPError(
        TOOL_EXECUTION_ERROR,
        error.safe_message,
        envelope.model_dump(mode="json"),
    )


class ConfigurationError(TestAutomationSDKError):
    """Raised when runtime configuration is missing or invalid."""


class ArtifactError(TestAutomationSDKError):
    """Raised when a persisted artifact cannot be read or validated."""


class EmbeddingError(TestAutomationSDKError):
    """Raised when an embedding provider fails or returns invalid data."""


class IndexBuildError(TestAutomationSDKError):
    """Raised when documentation ingestion or index publication fails."""


class RetrievalError(TestAutomationSDKError):
    """Raised when a retrieval operation cannot produce a valid result."""
