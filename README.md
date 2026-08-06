# dSPACE TestAutomationSDK MCP Server

This package provides one read-only MCP tool, `retrieve_documentation`, backed
by a committed FAISS index of the TestAutomationSDK documentation. The server
communicates over MCP stdio. It does not crawl documentation or rebuild the
index at startup.

## Prerequisites

- Python 3.12, 3.13, or 3.14.
- An Ollama-compatible embedding endpoint.
- The exact `nomic-embed-text:v1.5` model, unless using a rebuilt index with a
	different explicitly configured model.
- `uv` for development, index maintenance, and wheel verification.

For a local Ollama installation, make sure the Ollama service is running and
pull the pinned model:

```powershell
ollama pull nomic-embed-text:v1.5
```

The server uses Ollama's `/api/embed` endpoint. `ollama pull` installs the
model. The MCP server connects to the configured service directly.

## Installation

### Repository Checkout

The simplest development and local MCP setup is to clone the repository and
let `uv` create and populate the project environment:

```text
git clone <repository-url>
cd TestAutomationSDKMCP
uv sync
```

On Windows, start the MCP server from the checkout with `SDKMCP.cmd`. On Linux
or macOS, make `SDKMCP.sh` executable once and use it to start the MCP server:

```text
./SDKMCP.sh
```

Both launchers run `uv --directory` against the repository directory, so they
work regardless of the caller's current working directory and forward all
arguments to the MCP server. `uv` must be installed and available on `PATH`.

### Global Wheel Installation

For a user who does not want a repository checkout, download the release
`*.whl` and install it as a globally callable `uv` tool:

```powershell
uv tool install .\test_automation_sdk_mcp-0.1.0-py3-none-any.whl
test-automation-sdk-mcp
```

On Linux or macOS, use the downloaded wheel path in the same command:

```text
uv tool install ./test_automation_sdk_mcp-0.1.0-py3-none-any.whl
test-automation-sdk-mcp
```

After installation, configure the MCP client to call
`test-automation-sdk-mcp` directly. If the command is not found, add the `uv`
tool executable directory reported by `uv tool update-shell` to `PATH`.

The wheel contains the package code and these three generated artifacts:

- `test_automation_sdk_mcp/db/TA_Docu.faiss`
- `test_automation_sdk_mcp/db/TA_Docu.documents.json`
- `test_automation_sdk_mcp/db/TA_Docu.manifest.json`

The runtime discovers the default artifact set through
`importlib.resources`. It does not use the current working directory. Set
`TA_SDK_DB_DIR` only when selecting an alternate, fully validated artifact
directory.

## MCP Client Configuration

Configure an MCP client with the installed console script. The exact JSON
location depends on the client. The stdio server entry has this shape:

```json
{
	"mcpServers": {
		"test-automation-sdk": {
			"command": "test-automation-sdk-mcp",
			"args": [],
			"env": {
				"TA_SDK_OLLAMA_URL": "http://127.0.0.1:11434",
				"TA_SDK_OLLAMA_MODEL": "nomic-embed-text:v1.5"
			}
		}
	}
}
```

On Windows, use the full path to
`<venv>\Scripts\test-automation-sdk-mcp.exe` when the virtual environment is
not active. The server writes protocol messages to stdout. Startup progress
and errors go to stderr.

For a repository checkout, use the absolute path to `SDKMCP.cmd` on Windows or
`SDKMCP.sh` on Linux/macOS as the MCP command instead. For a wheel installed
with `uv tool install`, use `test-automation-sdk-mcp` directly.

## Configuration

All settings are read when the server or index builder starts:

| Variable                 | Default                  | Description                                                  |
| ------------------------ | ------------------------ | ------------------------------------------------------------ |
| `TA_SDK_OLLAMA_URL`      | `http://127.0.0.1:11434` | Ollama-compatible base URL. The client appends `/api/embed`. |
| `TA_SDK_OLLAMA_MODEL`    | `nomic-embed-text:v1.5`  | Model used for query embeddings. It must match the manifest. |
| `TA_SDK_OLLAMA_API_KEY`  | unset                    | Optional bearer-token value for a managed endpoint.          |
| `TA_SDK_RESULT_COUNT`    | `5`                      | Number of nearest snippets, from 1 through 50.               |
| `TA_SDK_DB_DIR`          | packaged `db/`           | Optional filesystem artifact-directory override.             |
| `TA_SDK_CONNECT_TIMEOUT` | `5`                      | HTTP connection timeout in seconds.                          |
| `TA_SDK_REQUEST_TIMEOUT` | `30`                     | HTTP request timeout in seconds.                             |

When `TA_SDK_OLLAMA_API_KEY` is set, requests contain
`Authorization: Bearer <value>`. The key is never logged or included in error
messages. Do not put secrets in a committed client configuration file. Use the
client's environment or secret-management facility.

The text supplied to `retrieve_documentation` is sent as the `input` value in a
POST request to the configured Ollama endpoint. Do not send confidential query
text to an endpoint that is not approved for that data.

### Local Ollama

```powershell
$env:TA_SDK_OLLAMA_URL = "http://127.0.0.1:11434"
$env:TA_SDK_OLLAMA_MODEL = "nomic-embed-text:v1.5"
ollama pull nomic-embed-text:v1.5
test-automation-sdk-mcp
```

### Centrally Hosted Ollama-Compatible Endpoint

```powershell
$env:TA_SDK_OLLAMA_URL = "https://embeddings.example.test"
$env:TA_SDK_OLLAMA_MODEL = "nomic-embed-text:v1.5"
$env:TA_SDK_OLLAMA_API_KEY = "set-this-through-your-secret-store"
test-automation-sdk-mcp
```

The endpoint must accept `POST /api/embed`, return the requested model name
when supplied, and return one finite 768-value vector for each input.

## Maintainer Index Rebuild

The committed index was built from the complete `data/` tree with
`nomic-embed-text:v1.5`. Rebuild it only when the source documentation or
embedding model changes:

```powershell
$env:TA_SDK_OLLAMA_URL = "http://127.0.0.1:11434"
$env:TA_SDK_OLLAMA_MODEL = "nomic-embed-text:v1.5"
uv run test-automation-sdk-mcp-build-index `
	--source data `
	--output src/test_automation_sdk_mcp/db
```

The builder validates source coverage, chunks deterministically, embeds in
batches, validates all vectors, reloads the staged artifacts, and publishes the
three files together. The manifest records the model, 768 dimensions, L2
metric, document count, and SHA-256 hashes. Review all three generated files.

## Tests and Wheel Validation

Run the non-network checks first, then the marked tests with local Ollama:

```powershell
uv run ruff check src tests
uv run pyright
uv run pytest -m "not ollama"
uv run pytest -m ollama
uv build
```

Run the automated clean-wheel validation separately when reviewing a release:

```powershell
uv run pytest -m release
```

This test builds a wheel into a unique temporary directory, installs it into a
unique temporary virtual environment, validates the packaged artifacts from a
foreign working directory, and opens an MCP session through the installed
console script. MCP initialization would fail if the console script wrote
non-MCP text to stdout.

Inspect the wheel and verify that all three artifacts are present:

```powershell
uv run python -c "from zipfile import ZipFile; from pathlib import Path; wheel = next(Path('dist').glob('*.whl')); print(*[name for name in ZipFile(wheel).namelist() if '/db/' in name], sep='\n')"
```

For a clean-install smoke test, create the environment outside the repository
and run the console script from a different working directory:

```powershell
$wheel = (Resolve-Path .\dist\test_automation_sdk_mcp-0.1.0-py3-none-any.whl).Path
$releaseVenv = Join-Path $env:TEMP "test-automation-sdk-mcp-wheel"
uv venv $releaseVenv
uv pip install --python "$releaseVenv\Scripts\python.exe" $wheel
Push-Location $env:TEMP
try {
	& "$releaseVenv\Scripts\python.exe" -c "from test_automation_sdk_mcp.index import load_packaged_artifacts; a = load_packaged_artifacts(); assert a.manifest.document_count == a.index.ntotal"
} finally {
	Pop-Location
}
```

The MCP stdio smoke test should initialize a client session and list exactly
one tool, `retrieve_documentation`, while capturing stdout and confirming it
contains no non-MCP text. Use the same clean environment and configure
`TA_SDK_DB_DIR` only if testing an alternate artifact directory.

## Troubleshooting

**Endpoint or connection errors**

- Set `TA_SDK_OLLAMA_URL` to the base URL, without `/api/embed`.
- Confirm the endpoint is reachable and accepts the Ollama embed contract.
- Increase `TA_SDK_CONNECT_TIMEOUT` or `TA_SDK_REQUEST_TIMEOUT` for a remote
	service, within the supported 300-second maximum.

**Model mismatch**

- `TA_SDK_OLLAMA_MODEL` must match `embedding_model` in
	`TA_Docu.manifest.json`.
- Pull the exact pinned model or rebuild the index with the model you intend
	to operate. The server refuses to start on a mismatch.

**Dimension or response errors**

- The configured embedding service must return finite vectors with exactly 768
	values for this artifact set.
- Check the service response model, input count, and model installation. Do
	not edit the FAISS or JSON files by hand.

**Missing, mixed, or corrupted artifacts**

- Keep `TA_Docu.faiss`, `TA_Docu.documents.json`, and
	`TA_Docu.manifest.json` from the same builder run.
- Check `TA_SDK_DB_DIR` for a typo and ensure all three files are readable.
- The loader verifies manifest counts, FAISS type and dimension, document
	counts, and artifact hashes before serving requests.
- The legacy root `db/TA_Docu.pkl` is not used and must not be packaged.

**Unexpected console output**

MCP messages use stdout. Redirect diagnostics to stderr and ensure the client
starts `test-automation-sdk-mcp` directly rather than through a wrapper that
prints banners or progress text.
