# dSPACE TestAutomationSDK MCP Server

This package provides one read-only MCP tool, `retrieve_documentation`, backed
by a committed FAISS index of the TestAutomationSDK documentation. The server
communicates over MCP stdio. It does not crawl documentation or rebuild the
index at startup.

## Prerequisites

- Python 3.12, 3.13, or 3.14.
- An Ollama-compatible or OpenAI-compatible embedding endpoint.
- The packaged index was built with Ollama and
	`nomic-embed-text:v1.5`. An OpenAI-compatible endpoint can query that same
	index; embedding-space compatibility is an operator responsibility.
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

### Release Assets and Checksums

Pushing a tag matching `v*` runs the release workflow. Each GitHub release
publishes the wheel, source `tar.gz`, a release zip, and `SHA256SUMS.txt`. The
release zip contains the wheel, source archive, and `license.txt` generated
from the project's installed dependencies.

On Linux or macOS, download the release assets into the same directory and
verify them with:

```text
sha256sum -c SHA256SUMS.txt
```

On Windows, calculate an individual asset's SHA-256 value with PowerShell:

```powershell
Get-FileHash .\test_automation_sdk_mcp-0.1.0-py3-none-any.whl -Algorithm SHA256
```

Compare the result with the corresponding entry in `SHA256SUMS.txt`.
Checksums detect corruption or altered downloads; signed releases or artifact
attestations are needed when publisher authenticity must also be verified.

## MCP Client Configuration

Configure an MCP client with the installed console script. The exact JSON
location depends on the client. Choose one of these provider configurations.

For the default local Ollama provider:

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

For the verified local llama.cpp OpenAI-compatible endpoint:

```json
{
	"mcpServers": {
		"test-automation-sdk": {
			"command": "test-automation-sdk-mcp",
			"args": [],
			"env": {
				"TA_SDK_EMBEDDING_PROVIDER": "openai",
				"TA_SDK_OPENAI_URL": "http://127.0.0.1:8080/v1/embeddings",
				"TA_SDK_OPENAI_MODEL": "nomic-embed-text-v1.5.Q8_0.gguf"
			}
		}
	}
}
```

For a repository checkout, use the absolute path to `SDKMCP.cmd` on Windows or
`SDKMCP.sh` on Linux/macOS as the MCP command instead. For a wheel installed
with `uv tool install`, use `test-automation-sdk-mcp` directly.

## Configuration

All settings are read when the server or index builder starts:

| Variable                    | Default                  | Description                                                        |
| --------------------------- | ------------------------ | ------------------------------------------------------------------ |
| `TA_SDK_EMBEDDING_PROVIDER` | `ollama`                 | `ollama` or `openai`, case-insensitive.                            |
| `TA_SDK_OLLAMA_URL`         | `http://127.0.0.1:11434` | Ollama-compatible base URL; the client appends `/api/embed`.       |
| `TA_SDK_OLLAMA_MODEL`       | `nomic-embed-text:v1.5`  | Required Ollama request model; manifest value is build provenance. |
| `TA_SDK_OLLAMA_API_KEY`     | unset                    | Optional Ollama bearer-token value.                                |
| `TA_SDK_OPENAI_URL`         | unset                    | Required complete endpoint URL ending in `/v1/embeddings`.         |
| `TA_SDK_OPENAI_MODEL`       | unset                    | Optional endpoint model identifier; not an artifact admission key. |
| `TA_SDK_OPENAI_API_KEY`     | unset                    | Optional OpenAI-compatible bearer-token value.                     |
| `TA_SDK_RESULT_COUNT`       | `5`                      | Number of nearest snippets, from 1 through 50.                     |
| `TA_SDK_DB_DIR`             | packaged `db/`           | Optional filesystem artifact-directory override.                   |
| `TA_SDK_CONNECT_TIMEOUT`    | `5`                      | HTTP connection timeout in seconds.                                |
| `TA_SDK_REQUEST_TIMEOUT`    | `30`                     | HTTP request timeout in seconds.                                   |

When `TA_SDK_OLLAMA_API_KEY` is set, requests contain
`Authorization: Bearer <value>`. The key is never logged or included in error
messages. Do not put secrets in a committed client configuration file. Use the
client's environment or secret-management facility.

The text supplied to `retrieve_documentation` and the documentation text sent
while building an index are transmitted to the selected embedding endpoint.
The provider and model recorded in the manifest describe how document vectors
were built. Runtime does not reject a different provider or model; the operator
owns the risk of incompatible pooling, normalization, task prefixes, model
conversion, quantization, or backend settings.

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

### Local OpenAI-Compatible Endpoint

The OpenAI-compatible URL is the complete embeddings endpoint. Set the model
when the endpoint supports or requires explicit model selection:

```powershell
$env:TA_SDK_EMBEDDING_PROVIDER = "openai"
$env:TA_SDK_OPENAI_URL = "http://127.0.0.1:8080/v1/embeddings"
$env:TA_SDK_OPENAI_MODEL = "nomic-embed-text-v1.5.Q8_0.gguf"
test-automation-sdk-mcp
```

The packaged Ollama-built index is used automatically; these provider settings
are all that is required. If the endpoint always serves one active model, omit
`TA_SDK_OPENAI_MODEL` and the request will omit its `model` field.

### Authenticated Remote OpenAI-Compatible Endpoint

```powershell
$env:TA_SDK_EMBEDDING_PROVIDER = "openai"
$env:TA_SDK_OPENAI_URL = "https://embeddings.example.test/v1/embeddings"
$env:TA_SDK_OPENAI_MODEL = "approved-embedding-model"
$env:TA_SDK_OPENAI_API_KEY = "set-this-through-your-secret-store"
test-automation-sdk-mcp
```

If `TA_SDK_OPENAI_MODEL` is omitted, the model field is omitted from requests.
If supplied, the provider may validate a returned response model against that
request value, but neither form is compared with the packaged manifest.

### Verified llama.cpp Reference

This measured configuration used the packaged Ollama-built index:

```powershell
llama-server `
	-m nomic-embed-text-v1.5.Q8_0.gguf `
	--embedding `
	--host 0.0.0.0 `
	--port 8080 `
	-c 8192 `
	-b 8192 `
	-ub 8192 `
	--rope-scaling yarn `
	--rope-freq-scale 0.75
```

On the committed 12-query and 128-document probe corpus, both endpoints
returned finite 768-dimensional vectors. A reference run measured same-input
cosine p5 `0.99654`, pairwise geometry correlation `0.99815`, query top-5
overlap `0.98333`, and document top-10 neighborhood overlap `0.95781`. Values
may vary slightly across backend versions; these results apply only to the
tested model file and server settings.

### Optional Compatibility Check

Run the advisory comparison before trying a different model, quantization,
conversion, pooling, normalization, task prefix, or backend configuration.
Both Ollama and the candidate OpenAI-compatible endpoint must be running.

From a repository checkout, this self-contained command uses the default local
Ollama service and the verified llama.cpp endpoint:

```powershell
uv run test-automation-sdk-mcp-check-embedding-compatibility `
	--openai-url http://127.0.0.1:8080/v1/embeddings `
	--openai-model nomic-embed-text-v1.5.Q8_0.gguf
```

With a global wheel installation, omit `uv run`. The command also reads the
normal `TA_SDK_OLLAMA_*` and `TA_SDK_OPENAI_*` variables; command-line URL and
model options override them. Add `--json` for machine-readable output.

Success requires finite 768-dimensional vectors, same-input cosine p5 at least
`0.995`, pairwise geometry correlation at least `0.995`, mean query top-5
overlap at least `0.95`, mean document top-10 overlap at least `0.95`, and
representative result checks. The command returns a nonzero exit code when the
advisory check fails. Runtime does not consume or enforce its report, and an
endpoint may still be tried after a failed result. JSON output contains
metrics, hashes, thresholds, and sampled row IDs, but no endpoints, keys,
request text, vectors, responses, or paths.

## Maintainer Index Rebuild

Before rebuilding, obtain the latest generated HTML documentation export,
including its `search.json`, and copy the complete tree into a new `data/`
folder at the repository root. Preserve all relative paths; the builder needs
the HTML pages and `search.json` to have matching coverage:

```powershell
New-Item -ItemType Directory -Force .\data | Out-Null
Copy-Item -Path C:\path\to\latest\html-docs\* -Destination .\data -Recurse -Force
```

The `data/` folder is ignored by Git and is only a local input to the index
builder. Do not commit it.

The committed index was built from the complete `data/` tree with the default
Ollama provider and `nomic-embed-text:v1.5`. Rebuild it when the source
documentation or document embedding configuration changes:

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
One packaged Ollama-built database is used by both providers. A separately
rebuilt index records its build provider and model as provenance, but runtime
admission does not require those values to match the query provider.

## Tests and Wheel Validation

Run the non-network checks first, then the marked tests with local Ollama:

```powershell
uv run ruff check src tests
uv run pyright
uv run pytest -m "not ollama and not openai and not release"
uv run pytest -m compatibility
uv run pytest -m ollama
uv run pytest -m openai
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

Expected `retrieve_documentation` failures are native MCP/JSON-RPC errors with
JSON-RPC code `-32000`, a safe human-readable message, and this versioned
`data` envelope:

```json
{
	"schema_version": 1,
	"code": "embedding_service_unavailable",
	"classification": "transient",
	"retryable": true
}
```

| Application code                     | Classification | Meaning                                                   |
| ------------------------------------ | -------------- | --------------------------------------------------------- |
| `invalid_query`                      | permanent      | The query is empty, oversized, or otherwise invalid.      |
| `embedding_request_timed_out`        | transient      | The embedding request exceeded its timeout.               |
| `embedding_service_unavailable`      | transient      | The embedding endpoint could not be reached.              |
| `embedding_service_rejected_request` | permanent      | The endpoint rejected a non-retryable request.            |
| `embedding_service_failure`          | transient      | The endpoint returned a retryable failure.                |
| `embedding_response_invalid`         | permanent      | The response violates the embedding contract.             |
| `retrieval_failed`                   | permanent      | The index or search result violates retrieval invariants. |
| `tool_output_invalid`                | permanent      | A snippet violates the public success schema.             |
| `server_runtime_unavailable`         | transient      | The server lifecycle context is temporarily unavailable.  |

Retry only errors classified as `transient`, using bounded backoff. For a
`permanent` error, change the query or configuration before retrying. Startup
configuration and artifact failures occur before an MCP session exists; they
are reported on stderr and are not MCP tool errors.

**Endpoint or connection errors**

- Set `TA_SDK_OLLAMA_URL` to the base URL, without `/api/embed`.
- Confirm the endpoint is reachable and accepts the Ollama embed contract.
- Increase `TA_SDK_CONNECT_TIMEOUT` or `TA_SDK_REQUEST_TIMEOUT` for a remote
	service, within the supported 300-second maximum.

**Embedding compatibility**

- The manifest provider and model are build provenance, not runtime admission
	keys. A differing provider or model does not prevent startup.
- Run the optional compatibility checker when changing quantization, model
	conversion, pooling, normalization, task prefixes, or backend settings.
- A failed advisory check does not block runtime, but it indicates retrieval
	quality risk that the operator must accept or resolve.

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
