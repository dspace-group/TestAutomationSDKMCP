# dSPACE Test Automation SDK MCP Server

This server provides the `retrieve_documentation` tool for the Test Automation
SDK, backed by a committed index of the Test Automation SDK documentation.

## Quick Start

This chapter is for users who want to install the MCP server and connect it to
an MCP client. For all environment variables, hosted endpoints, and provider
details, see [Configuration](#configuration). For deployment and compatibility
checks, see [Operations](#operations).

### Prerequisites

- Python 3.12, 3.13, or 3.14.
- An Ollama-compatible or OpenAI-compatible embedding endpoint.
- `uv` for installing the published wheel or running the repository checkout.

> [!CAUTION]
> Both local Ollama setup options below download and run
> `nomic-embed-text:v1.5` from Ollama. Third-party model artifacts may
> introduce supply-chain, licensing, privacy, and security risks. Review and
> approve the model source and terms according to your organization's policies
> before proceeding.

For a local Ollama installation, make sure the Ollama service is running and
pull the pinned model:

```sh
ollama pull nomic-embed-text:v1.5
```

Alternatively, the repository's Docker Compose configuration starts Ollama in
a container and downloads the pinned model into a persistent Docker volume.
From the repository root, run:

```sh
docker compose -f docker/compose.yaml up -d
```

### Choose an installation option

#### Option 1: Install the published wheel

Download the release `*.whl` and install it as a globally callable `uv` tool:

```sh
uv tool install .\test_automation_sdk_mcp-0.1.0-py3-none-any.whl
```

On Linux or macOS:

```sh
uv tool install ./test_automation_sdk_mcp-0.1.0-py3-none-any.whl
```

If the command is not found after installation, add the `uv` tool executable
directory reported by `uv tool update-shell` to `PATH`.

#### Option 2: Clone the repository and run it with `uv`

Clone the repository and create its environment:

```text
git clone <repository-url>
cd TestAutomationSDKMCP
uv sync
```

The repository includes launchers that run the server with the repository's
environment. On Windows, use `SDKMCP.cmd`. On Linux or macOS, make
`SDKMCP.sh` executable once and use it:

```sh
chmod +x SDKMCP.sh
./SDKMCP.sh
```

The launchers find the repository relative to their own location, so they work
regardless of the caller's current directory and forward arguments to the MCP
server. When configuring an MCP client, use the absolute path to the launcher
as its command:

- Windows: `C:\\path\\to\\TestAutomationSDKMCP\\SDKMCP.cmd`
- Linux/macOS: `/path/to/TestAutomationSDKMCP/SDKMCP.sh`

### Configure the MCP client

Configure the client to call the command from the installation option you
selected. The exact JSON location depends on the client. For a wheel
installation, use `test-automation-sdk-mcp` directly. For a repository
checkout, use the absolute path to `SDKMCP.cmd` or `SDKMCP.sh` described above.
This is the default local Ollama configuration for a wheel installation:

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

Restart or reload the MCP client and confirm that it lists one tool named
`retrieve_documentation`. The packaged documentation index is included
automatically and standard installations do not need `TA_SDK_DB_DIR`.

For an OpenAI-compatible endpoint, use the configuration shown in
[OpenAI-compatible providers](#openai-compatible-providers). For a checkout
based setup, see [Repository checkout](#repository-checkout).

## Configuration

All settings are read when the server or index builder starts. The default
configuration is suitable for a local Ollama service; most users only need to
set the endpoint and model for their embedding service.

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

### Ollama providers

#### Local Ollama

```sh
$env:TA_SDK_OLLAMA_URL = "http://127.0.0.1:11434"
$env:TA_SDK_OLLAMA_MODEL = "nomic-embed-text:v1.5"
ollama pull nomic-embed-text:v1.5
test-automation-sdk-mcp
```

The Docker Compose setup publishes its Ollama endpoint on loopback at
`http://127.0.0.1:11434` by default. If port `11434` is already in use, select
another loopback port before starting the stack by setting the
`TA_SDK_OLLAMA_PORT` environment variable:

```powershell
$env:TA_SDK_OLLAMA_PORT = "11435"
docker compose -f docker/compose.yaml up -d
```

In that case, configure the MCP server with
`TA_SDK_OLLAMA_URL=http://127.0.0.1:11435`. To stop Ollama and remove the
containers and network while retaining the downloaded model volume, run:

```sh
docker compose -f docker/compose.yaml down
```

#### Centrally hosted Ollama-compatible endpoint

```sh
$env:TA_SDK_OLLAMA_URL = "https://embeddings.example.test"
$env:TA_SDK_OLLAMA_MODEL = "nomic-embed-text:v1.5"
$env:TA_SDK_OLLAMA_API_KEY = "set-this-through-your-secret-store"
test-automation-sdk-mcp
```

The endpoint must accept `POST /api/embed`, return the requested model name
when supplied, and return one finite 768-value vector for each input.

### OpenAI-compatible providers

#### Local OpenAI-compatible endpoint

The OpenAI-compatible URL is the complete embeddings endpoint. Set the model
when the endpoint supports or requires explicit model selection:

```sh
$env:TA_SDK_EMBEDDING_PROVIDER = "openai"
$env:TA_SDK_OPENAI_URL = "http://127.0.0.1:8080/v1/embeddings"
$env:TA_SDK_OPENAI_MODEL = "nomic-embed-text-v1.5.Q8_0.gguf"
test-automation-sdk-mcp
```

The packaged Ollama-built index is used automatically; these provider settings
are all that is required. If the endpoint always serves one active model, omit
`TA_SDK_OPENAI_MODEL` and the request will omit its `model` field.

#### Authenticated remote OpenAI-compatible endpoint

```sh
$env:TA_SDK_EMBEDDING_PROVIDER = "openai"
$env:TA_SDK_OPENAI_URL = "https://embeddings.example.test/v1/embeddings"
$env:TA_SDK_OPENAI_MODEL = "approved-embedding-model"
$env:TA_SDK_OPENAI_API_KEY = "set-this-through-your-secret-store"
test-automation-sdk-mcp
```

If `TA_SDK_OPENAI_MODEL` is omitted, the model field is omitted from requests.
If supplied, the provider may validate a returned response model against that
request value, but neither form is compared with the packaged manifest.

#### Verified llama.cpp reference

This measured configuration used the packaged Ollama-built index and can be
used as OpenAI-compatible endpoint instead of Ollama:

```sh
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

## Operations

This chapter is for administrators and users operating a non-default provider
or deployment. Runtime uses the packaged index and does not crawl documents or
rebuild the index during startup.

### Packaged and alternate index artifacts

The wheel contains the package code and these three generated artifacts:

- `test_automation_sdk_mcp/db/TA_Docu.faiss`
- `test_automation_sdk_mcp/db/TA_Docu.documents.json`
- `test_automation_sdk_mcp/db/TA_Docu.manifest.json`

The default artifact set is loaded from the installed package rather than the
current working directory. Set `TA_SDK_DB_DIR` only when selecting an
alternate directory containing a complete, validated artifact set.

### Release verification

Download the release assets into the same directory and verify their checksums.
On Linux or macOS:

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

### Embedding compatibility checks

The compatibility command is advisory and has two explicit modes. The default
`index` mode evaluates one configured candidate endpoint against the exact
verified documentation index. It does not start or contact an Ollama baseline;
the stored document vectors are the reference, and deterministic query probes
check representative documentation locations.

For an OpenAI-compatible candidate, select the provider as usual and run:

```sh
$env:TA_SDK_EMBEDDING_PROVIDER = "openai"
$env:TA_SDK_OPENAI_URL = "http://127.0.0.1:8080/v1/embeddings"
$env:TA_SDK_OPENAI_MODEL = "nomic-embed-text-v1.5.Q8_0.gguf"
uv run test-automation-sdk-mcp-check-embedding-compatibility --json
```

The command also reads the normal `TA_SDK_OLLAMA_*` and `TA_SDK_OPENAI_*`
variables. Provider-specific URL and model options override those values for
the selected role. Index mode reports structural checks, candidate-versus-index
document metrics, representative results, artifact hashes, thresholds, and
sampled row IDs.

Use live `parity` mode when the question is whether two active providers return
similar vectors for the same probe inputs. This mode requires both providers;
the example below compares the default local Ollama baseline with the verified
OpenAI-compatible endpoint:

```sh
uv run test-automation-sdk-mcp-check-embedding-compatibility `
    --mode parity `
    --openai-url http://127.0.0.1:8080/v1/embeddings `
    --openai-model nomic-embed-text-v1.5.Q8_0.gguf `
    --json
```

Parity mode retains same-input cosine, pairwise geometry, query-neighbor
overlap, document-neighborhood overlap, and representative result checks.
Both modes return a nonzero exit code when their applicable advisory checks
fail. Runtime does not consume or enforce either report, and an endpoint may
still be tried after a failed result. JSON output identifies the mode and
contains only safe metrics, hashes, thresholds, and sampled row IDs; it never
contains endpoints, keys, request text, vectors, responses, or local paths.

## Development and Maintenance

This chapter is for contributors and maintainers. Normal users do not need to
clone the repository, rebuild the index, or run the test suite.

### Repository checkout

Clone the repository and let `uv` create and populate the project environment:

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

When configuring an MCP client with a checkout, use the absolute path to
`SDKMCP.cmd` on Windows or `SDKMCP.sh` on Linux/macOS as the MCP command.

### Rebuild the documentation index

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

```sh
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

### Tests and wheel validation

Run the non-network checks first, then the marked tests with local Ollama:

```sh
uv run ruff check src tests
uv run pyright
uv run pytest -m "not ollama and not openai and not release"
uv run pytest -m compatibility
uv run pytest -m ollama
uv run pytest -m openai
uv build
```

Run the automated clean-wheel validation separately when reviewing a release:

```sh
uv run pytest -m release
```

This test builds a wheel into a unique temporary directory, installs it into a
unique temporary virtual environment, validates the packaged artifacts from a
foreign working directory, and opens an MCP session through the installed
console script. MCP initialization would fail if the console script wrote
non-MCP text to stdout.

Inspect the wheel and verify that all three artifacts are present:

```sh
uv run python -c "from zipfile import ZipFile; from pathlib import Path; wheel = next(Path('dist').glob('*.whl')); print(*[name for name in ZipFile(wheel).namelist() if '/db/' in name], sep='\n')"
```

For a clean-install smoke test, create the environment outside the repository
and run the console script from a different working directory:

```sh
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

### Release workflow

Pushing a tag matching `v*` runs the release workflow. Each GitHub release
publishes the wheel, source `tar.gz`, a release zip, and `SHA256SUMS.txt`. The
release zip contains the wheel, source archive, and `license.txt` generated
from the project's installed dependencies.

## Troubleshooting

For expected retrieval failures, the server returns a native MCP/JSON-RPC
error with JSON-RPC code `-32000`, a safe human-readable message, and this
versioned `data` envelope:

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

## Support

For technical questions and issues related to the dSPACE MCP Servers and related
GitHub repositories, please open a GitHub issue.

As a valued dSPACE customer, you are always welcome to contact dSPACE Support
directly via http://www.dspace.com/go/supportrequest.
