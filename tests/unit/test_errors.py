import pytest
from pydantic import ValidationError

from test_automation_sdk_mcp.errors import (
    ERROR_CLASSIFICATIONS,
    TOOL_EXECUTION_ERROR,
    ApplicationErrorCode,
    ArtifactError,
    ConfigurationError,
    EmbeddingError,
    ErrorClassification,
    PublicErrorEnvelope,
    RetrievalError,
    TestAutomationSDKError,
    to_mcp_error,
)


def test_project_errors_have_one_safe_root_and_distinct_categories() -> None:
    errors = (ConfigurationError, ArtifactError, EmbeddingError, RetrievalError)

    assert all(issubclass(error_type, TestAutomationSDKError) for error_type in errors)
    error = ArtifactError("internal response body", safe_message="Artifact is unavailable.")
    assert str(error) == "Artifact is unavailable."
    assert error.safe_message == "Artifact is unavailable."


def test_every_public_code_has_exactly_one_classification() -> None:
    assert set(ERROR_CLASSIFICATIONS) == set(ApplicationErrorCode)
    assert all(isinstance(value, ErrorClassification) for value in ERROR_CLASSIFICATIONS.values())


@pytest.mark.parametrize("code", ApplicationErrorCode)
def test_error_envelope_derives_retryable_from_classification(code: ApplicationErrorCode) -> None:
    envelope = PublicErrorEnvelope.for_code(code)

    assert envelope.schema_version == 1
    assert envelope.retryable is (envelope.classification is ErrorClassification.TRANSIENT)


def test_error_envelope_is_strict_and_forbids_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        PublicErrorEnvelope.model_validate(
            {
                "schema_version": 1,
                "code": "invalid_query",
                "classification": "permanent",
                "retryable": False,
                "unexpected": True,
            }
        )

    with pytest.raises(ValidationError):
        PublicErrorEnvelope.model_validate(
            {
                "schema_version": 2,
                "code": "invalid_query",
                "classification": "permanent",
                "retryable": False,
            }
        )

    with pytest.raises(ValidationError):
        PublicErrorEnvelope(
            code=ApplicationErrorCode.INVALID_QUERY,
            classification=ErrorClassification.TRANSIENT,
            retryable=True,
        )


def test_coded_exception_converts_to_safe_native_mcp_error() -> None:
    error = EmbeddingError(
        "API key secret at C:\\private\\endpoint with response body and stack trace",
        safe_message="Embedding service is unavailable.",
        code=ApplicationErrorCode.EMBEDDING_SERVICE_UNAVAILABLE,
    )

    converted = to_mcp_error(error)

    assert converted.code == TOOL_EXECUTION_ERROR
    assert converted.message == "Embedding service is unavailable."
    assert converted.data == {
        "schema_version": 1,
        "code": "embedding_service_unavailable",
        "classification": "transient",
        "retryable": True,
    }
    assert "secret" not in str(converted.error)
