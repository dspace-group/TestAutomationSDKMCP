import asyncio
import json
from collections.abc import Sequence
from pathlib import Path

import faiss
import numpy as np
from numpy.typing import NDArray

from test_automation_sdk_mcp.compatibility import (
    THRESHOLDS,
    CompatibilityProbe,  # pyright: ignore[reportPrivateUsage]
    ProbeCorpus,  # pyright: ignore[reportPrivateUsage]
    _inspect_vectors,  # pyright: ignore[reportPrivateUsage]
    _mean_overlap,  # pyright: ignore[reportPrivateUsage]
    _metric_checks,  # pyright: ignore[reportPrivateUsage]
    _neighborhood_overlap,  # pyright: ignore[reportPrivateUsage]
    run_compatibility_check,
)
from test_automation_sdk_mcp.documents import ChunkingManifest, DocumentRecord, DocumentStore, IndexManifest
from test_automation_sdk_mcp.index import ArtifactPaths, LoadedArtifacts


class StaticProvider:
    def __init__(self, vectors: NDArray[np.float32]) -> None:
        self.vectors = vectors

    async def embed(self, inputs: Sequence[str]) -> NDArray[np.float32]:
        assert len(inputs) == len(self.vectors)
        return self.vectors

    async def aclose(self) -> None:
        return None


def _artifacts(tmp_path: Path, vectors: NDArray[np.float32]) -> LoadedArtifacts:
    index = faiss.IndexFlatL2(768)
    index.add(vectors)  # pyright: ignore[reportUnknownMemberType]
    documents = DocumentStore(
        schema_version=1,
        documents=tuple(
            DocumentRecord(
                id=f"doc-{row_id}",
                location=f"concepts/topic-{row_id}.html",
                title=f"Topic {row_id}",
                breadcrumbs=("Concepts",),
                tags=(),
                chunk_index=0,
                content=f"Topic content {row_id}",
            )
            for row_id in range(len(vectors))
        ),
    )
    manifest = IndexManifest(
        schema_version=2,
        index_type="IndexFlatL2",
        distance_metric="l2",
        embedding_provider="ollama",
        embedding_model="baseline",
        embedding_dimension=768,
        document_count=len(vectors),
        search_json_sha256="0" * 64,
        html_tree_sha256="1" * 64,
        faiss_sha256="2" * 64,
        documents_sha256="3" * 64,
        chunking=ChunkingManifest(max_characters=1000, overlap_characters=200),
    )
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}", encoding="utf-8")
    return LoadedArtifacts(index, documents, manifest, ArtifactPaths(tmp_path))


def _corpus() -> ProbeCorpus:
    return ProbeCorpus(
        schema_version=1,
        queries=(
            CompatibilityProbe(
                id="topic-0",
                text="query zero",
                expected_locations=("concepts/topic-0.html",),
            ),
            CompatibilityProbe(
                id="topic-1",
                text="query one",
                expected_locations=("concepts/topic-1.html",),
            ),
        ),
        document_row_ids=tuple(range(12)),
    )


def _vectors() -> NDArray[np.float32]:
    values = np.zeros((12, 768), dtype=np.float32)
    for row_id in range(12):
        values[row_id, row_id] = 1.0
    return values


def test_identical_embedding_spaces_pass_and_report_is_redacted(tmp_path: Path) -> None:
    vectors = _vectors()
    report = asyncio.run(
        run_compatibility_check(
            StaticProvider(np.vstack((vectors[:2], vectors))),
            StaticProvider(np.vstack((vectors[:2], vectors))),
            _artifacts(tmp_path, vectors),
            _corpus(),
            corpus_sha256="a" * 64,
            artifact_hashes={"manifest_sha256": "b" * 64, "faiss_sha256": "c" * 64, "documents_sha256": "d" * 64},
        )
    )

    assert report["passed"] is True
    report_text = json.dumps(report, sort_keys=True)
    assert "query zero" not in report_text
    assert "http://secret-endpoint" not in report_text
    assert "api-key" not in report_text
    assert "vectors" not in report_text
    assert report["sampled_row_ids"] == list(range(12))


def test_transformed_candidate_fails_compatibility(tmp_path: Path) -> None:
    vectors = _vectors()
    candidate = np.vstack((vectors[:2], -vectors))
    report = asyncio.run(
        run_compatibility_check(
            StaticProvider(np.vstack((vectors[:2], vectors))),
            StaticProvider(candidate),
            _artifacts(tmp_path, vectors),
            _corpus(),
            artifact_hashes={"manifest_sha256": "b" * 64, "faiss_sha256": "c" * 64, "documents_sha256": "d" * 64},
        )
    )

    assert report["passed"] is False
    metric_checks = report["metric_checks"]
    assert isinstance(metric_checks, dict)
    assert metric_checks["same_input_cosine_p5"] is False


def test_metric_thresholds_are_inclusive() -> None:
    metrics: dict[str, float | int | None | bool] = {name: threshold for name, threshold in THRESHOLDS.items()}
    assert _metric_checks(metrics) == {name: True for name in THRESHOLDS}
    assert _mean_overlap((tuple(range(20)),), (tuple(range(19)) + (20,),), 20) == 0.95
    assert _neighborhood_overlap(({*range(19)},), ({*range(20)},), 20) == 0.95


def test_vector_inspection_rejects_wrong_shape_nonfinite_and_wrong_dtype() -> None:
    assert not _inspect_vectors(np.zeros((1, 768), dtype=np.float32), 2).structural_passed
    assert not _inspect_vectors(np.full((2, 768), np.nan, dtype=np.float32), 2).structural_passed
    assert not _inspect_vectors(np.ones((2, 768), dtype=np.float64), 2).structural_passed
