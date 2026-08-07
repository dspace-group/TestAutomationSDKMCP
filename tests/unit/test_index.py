import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

import faiss
import numpy as np
import pytest
from numpy.typing import NDArray

from test_automation_sdk_mcp.build_index import build_index
from test_automation_sdk_mcp.config import EmbeddingProviderKind
from test_automation_sdk_mcp.documents import ChunkingManifest, DocumentRecord, DocumentStore, IndexManifest
from test_automation_sdk_mcp.errors import ArtifactError
from test_automation_sdk_mcp.index import load_verified_artifacts, validate_faiss_index
from test_automation_sdk_mcp.provider import EmbeddingProfile


class FakeEmbeddingProvider:
    @property
    def profile(self) -> EmbeddingProfile:
        return EmbeddingProfile(EmbeddingProviderKind.OLLAMA, "fake")

    async def embed(self, inputs: Sequence[str]) -> NDArray[np.float32]:
        return np.zeros((len(inputs), 768), dtype=np.float32)


def make_source(root: Path) -> Path:
    source = root / "data"
    source.mkdir()
    (source / "index.html").write_text("raw html", encoding="utf-8")
    (source / "search.json").write_text(
        json.dumps(
            {
                "items": [
                    {
                        "location": "index.html",
                        "level": 1,
                        "title": "Intro",
                        "text": "<p>Body</p>",
                        "path": [],
                        "tags": [],
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return source


def test_verified_loader_rejects_tampered_documents(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    output = tmp_path / "db"
    asyncio.run(build_index(source, output, FakeEmbeddingProvider(), model="fake"))
    documents = output / "TA_Docu.documents.json"
    documents.write_bytes(documents.read_bytes() + b" ")

    with pytest.raises(ArtifactError, match="hash"):
        load_verified_artifacts(output)


def test_verified_loader_rejects_tampered_faiss(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    output = tmp_path / "db"
    asyncio.run(build_index(source, output, FakeEmbeddingProvider(), model="fake"))
    faiss_path = output / "TA_Docu.faiss"
    faiss_path.write_bytes(faiss_path.read_bytes() + b"tampered")

    with pytest.raises(ArtifactError, match="hash"):
        load_verified_artifacts(output)


@pytest.mark.parametrize("artifact_name", ["TA_Docu.faiss", "TA_Docu.documents.json", "TA_Docu.manifest.json"])
def test_verified_loader_rejects_missing_artifacts(tmp_path: Path, artifact_name: str) -> None:
    source = make_source(tmp_path)
    output = tmp_path / "db"
    asyncio.run(build_index(source, output, FakeEmbeddingProvider(), model="fake"))
    (output / artifact_name).unlink()

    with pytest.raises(ArtifactError, match="missing"):
        load_verified_artifacts(output)


def test_validate_faiss_rejects_wrong_type_dimension_and_count(tmp_path: Path) -> None:
    source = make_source(tmp_path)
    output = tmp_path / "db"
    asyncio.run(build_index(source, output, FakeEmbeddingProvider(), model="fake"))
    artifacts = load_verified_artifacts(output)

    with pytest.raises(ArtifactError, match="IndexFlatL2"):
        validate_faiss_index(faiss.IndexFlatIP(768), artifacts.manifest, artifacts.documents)
    with pytest.raises(ArtifactError, match="dimension"):
        validate_faiss_index(faiss.IndexFlatL2(384), artifacts.manifest, artifacts.documents)
    wrong_count_manifest = artifacts.manifest.model_copy(update={"document_count": 2})
    with pytest.raises(ArtifactError, match="counts"):
        validate_faiss_index(artifacts.index, wrong_count_manifest, artifacts.documents)


def test_validate_faiss_rejects_document_count_mismatch() -> None:
    digest = "a" * 64
    manifest = IndexManifest(
        schema_version=2,
        index_type="IndexFlatL2",
        distance_metric="l2",
        embedding_provider="ollama",
        embedding_model="fake",
        embedding_dimension=768,
        document_count=1,
        search_json_sha256=digest,
        html_tree_sha256=digest,
        faiss_sha256=digest,
        documents_sha256=digest,
        chunking=ChunkingManifest(max_characters=1000, overlap_characters=200),
    )
    store = DocumentStore(
        schema_version=1,
        documents=(
            DocumentRecord(
                id="one",
                location="index.html",
                title="One",
                breadcrumbs=(),
                tags=(),
                chunk_index=0,
                content="One",
            ),
        ),
    )
    index = faiss.IndexFlatL2(768)
    index.add(np.zeros((2, 768), dtype=np.float32))  # pyright: ignore[reportUnknownMemberType]

    with pytest.raises(ArtifactError, match="counts"):
        validate_faiss_index(index, manifest, store)
