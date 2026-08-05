"""Project-specific exceptions used at infrastructure boundaries."""


class TestAutomationSDKError(Exception):
    """Base class for errors that may be translated at a public boundary."""

    __test__ = False

    def __init__(self, message: str, *, safe_message: str | None = None) -> None:
        self.safe_message = safe_message if safe_message is not None else message
        super().__init__(self.safe_message)


class ConfigurationError(TestAutomationSDKError):
    """Raised when runtime configuration is missing or invalid."""


class ArtifactError(TestAutomationSDKError):
    """Raised when a persisted artifact cannot be read or validated."""


class EmbeddingError(TestAutomationSDKError):
    """Raised when an embedding provider fails or returns invalid data."""


class RetrievalError(TestAutomationSDKError):
    """Raised when a retrieval operation cannot produce a valid result."""
