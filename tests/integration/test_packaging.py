import os
import shutil
import subprocess
import textwrap
from collections.abc import Sequence
from pathlib import Path

import pytest

from test_automation_sdk_mcp.compatibility import packaged_probe_corpus
from test_automation_sdk_mcp.config import DEFAULT_MODEL
from test_automation_sdk_mcp.index import load_packaged_artifacts


def _run(command: Sequence[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True, capture_output=True, text=True, timeout=300)


def test_packaged_artifacts_load_outside_repository_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)

    artifacts = load_packaged_artifacts()

    assert artifacts.manifest.embedding_model == DEFAULT_MODEL
    assert artifacts.manifest.embedding_dimension == 768
    assert artifacts.manifest.distance_metric == "l2"
    assert artifacts.manifest.document_count == len(artifacts.documents.documents)
    assert artifacts.index.ntotal == artifacts.manifest.document_count
    assert artifacts.paths.directory != Path.cwd()
    with packaged_probe_corpus() as (_, corpus):
        assert len(corpus.queries) == 12
        assert len(corpus.document_row_ids) == 128


@pytest.mark.release
def test_clean_wheel_install_loads_artifacts_and_opens_stdio_session(tmp_path: Path) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    uv_executable = shutil.which("uv")
    if uv_executable is None:
        pytest.fail("uv is required for clean-wheel release validation")

    wheel_directory = tmp_path / "wheel"
    virtual_environment = tmp_path / "venv"
    _run([uv_executable, "build", "--wheel", "--out-dir", str(wheel_directory)], repository_root)
    wheels = list(wheel_directory.glob("*.whl"))
    assert len(wheels) == 1
    wheel = wheels[0]

    _run([uv_executable, "venv", str(virtual_environment)], repository_root)
    python_executable = virtual_environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    _run(
        [uv_executable, "pip", "install", "--python", str(python_executable), str(wheel)],
        repository_root,
    )

    probe_path = tmp_path / "clean_wheel_probe.py"
    probe_path.write_text(
        textwrap.dedent(
            """
            import asyncio
            import os
            import sys
            import tempfile
            from pathlib import Path

            import test_automation_sdk_mcp
            from mcp.client.session import ClientSession
            from mcp.client.stdio import StdioServerParameters, stdio_client

            from test_automation_sdk_mcp.index import load_packaged_artifacts


            async def main() -> None:
                module_path = Path(test_automation_sdk_mcp.__file__ or "").resolve()
                virtual_environment = Path(sys.executable).resolve().parent.parent
                assert module_path.is_relative_to(virtual_environment)

                artifacts = load_packaged_artifacts()
                assert artifacts.manifest.document_count == len(artifacts.documents.documents)
                assert artifacts.manifest.document_count == artifacts.index.ntotal

                environment = {
                    key: value
                    for key, value in os.environ.items()
                    if not key.startswith("TA_SDK_") and key != "PYTHONPATH"
                }
                executable_name = "test-automation-sdk-mcp.exe" if os.name == "nt" else "test-automation-sdk-mcp"
                parameters = StdioServerParameters(
                    command=str(Path(sys.executable).with_name(executable_name)),
                    args=[],
                    env=environment,
                    cwd=Path.cwd(),
                )
                with tempfile.TemporaryFile(mode="w+", encoding="utf-8") as stderr:
                    async with stdio_client(parameters, errlog=stderr) as (read_stream, write_stream):
                        async with ClientSession(read_stream, write_stream) as session:
                            await session.initialize()
                            tools = await session.list_tools()
                            assert [tool.name for tool in tools.tools] == ["retrieve_documentation"]
                    stderr.seek(0)
                    assert "error:" not in stderr.read().lower()


            asyncio.run(main())
            """
        ).strip()
        + "\n",
        encoding="utf-8",
    )
    environment = {key: value for key, value in os.environ.items() if key != "PYTHONPATH"}
    subprocess.run(
        [str(python_executable), str(probe_path)],
        cwd=tmp_path,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
        timeout=300,
    )
