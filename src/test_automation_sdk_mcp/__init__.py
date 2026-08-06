"""Console entry point for the Test Automation SDK MCP server."""

import logging
import sys

from .errors import TestAutomationSDKError
from .server import create_server


class _SingleLineFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        return message.replace("\r\n", r"\n").replace("\r", r"\r").replace("\n", r"\n")


def _configure_logging(log_level: str) -> None:
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_SingleLineFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logging.basicConfig(level=log_level, handlers=[handler], force=True)


def main() -> int:
    """Start the MCP server over stdio."""

    try:
        server = create_server()
        _configure_logging(server.settings.log_level)
        server.run(transport="stdio")
    except TestAutomationSDKError as error:
        print(f"error: {error.safe_message}", file=sys.stderr)
        return 1
    return 0


__all__ = ["main"]
