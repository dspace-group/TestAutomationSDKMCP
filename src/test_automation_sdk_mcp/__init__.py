"""Console entry point for the Test Automation SDK MCP server."""

import sys

from .errors import TestAutomationSDKError
from .server import create_server


def main() -> int:
    """Start the MCP server over stdio."""

    try:
        server = create_server()
        server.run(transport="stdio")
    except TestAutomationSDKError as error:
        print(f"error: {error.safe_message}", file=sys.stderr)
        return 1
    return 0


__all__ = ["main"]
