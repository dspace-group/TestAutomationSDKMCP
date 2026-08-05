from test_automation_sdk_mcp.errors import (
    ArtifactError,
    ConfigurationError,
    EmbeddingError,
    RetrievalError,
    TestAutomationSDKError,
)


def test_project_errors_have_one_safe_root_and_distinct_categories() -> None:
    errors = (ConfigurationError, ArtifactError, EmbeddingError, RetrievalError)

    assert all(issubclass(error_type, TestAutomationSDKError) for error_type in errors)
    error = ArtifactError("internal response body", safe_message="Artifact is unavailable.")
    assert str(error) == "Artifact is unavailable."
    assert error.safe_message == "Artifact is unavailable."
