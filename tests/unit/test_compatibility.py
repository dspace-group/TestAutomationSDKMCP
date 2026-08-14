import asyncio
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

import faiss
import numpy as np
import pytest
from numpy.typing import NDArray

import test_automation_sdk_mcp.compatibility as compatibility_module
from test_automation_sdk_mcp.compatibility import (
    INDEX_THRESHOLDS,
    THRESHOLDS,
    CompatibilityProbe,  # pyright: ignore[reportPrivateUsage]
    ProbeCorpus,  # pyright: ignore[reportPrivateUsage]
    _inspect_vectors,  # pyright: ignore[reportPrivateUsage]
    _mean_overlap,  # pyright: ignore[reportPrivateUsage]
    _metric_checks,  # pyright: ignore[reportPrivateUsage]
    _neighborhood_overlap,  # pyright: ignore[reportPrivateUsage]
    _parser,  # pyright: ignore[reportPrivateUsage]
    run_compatibility_check,
    run_index_compatibility_check,
)
from test_automation_sdk_mcp.documents import ChunkingManifest, DocumentRecord, DocumentStore, IndexManifest
from test_automation_sdk_mcp.errors import CompatibilityError
from test_automation_sdk_mcp.index import ArtifactPaths, LoadedArtifacts


class StaticProvider:
    def __init__(self, vectors: NDArray[np.float32]) -> None:
        self.vectors = vectors
        self.close_calls = 0

    async def embed(self, inputs: Sequence[str]) -> NDArray[np.float32]:
        assert len(inputs) == len(self.vectors)
        return self.vectors

    async def aclose(self) -> None:
        self.close_calls += 1


class RecordingProvider(StaticProvider):
    def __init__(self) -> None:
        super().__init__(np.ones((1, 768), dtype=np.float32))
        self.input_counts: list[int] = []

    async def embed(self, inputs: Sequence[str]) -> NDArray[np.float32]:
        self.input_counts.append(len(inputs))
        return np.ones((len(inputs), 768), dtype=np.float32)


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


def _varied_vectors() -> NDArray[np.float32]:
    values = np.random.default_rng(7).normal(size=(12, 768)).astype(np.float32)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / norms


def test_parser_defaults_to_index_and_supports_explicit_parity() -> None:
    assert _parser().parse_args([]).mode == "index"
    assert _parser().parse_args(["--mode", "parity"]).mode == "parity"


def test_index_composition_creates_and_closes_only_the_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    providers: list[RecordingProvider] = []

    def provider_factory(_config: object) -> RecordingProvider:
        provider = RecordingProvider()
        providers.append(provider)
        return provider

    monkeypatch.setenv("TA_SDK_EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("TA_SDK_OPENAI_URL", "https://embedding.example.test/v1/embeddings")
    monkeypatch.setattr(compatibility_module, "create_embedding_provider", provider_factory)

    report = asyncio.run(
        compatibility_module._check_from_arguments(  # pyright: ignore[reportPrivateUsage]
            compatibility_module._parser().parse_args(["--mode", "index"])  # pyright: ignore[reportPrivateUsage]
        )
    )

    assert report["mode"] == "index"
    assert len(providers) == 1
    assert providers[0].input_counts == [140]
    assert providers[0].close_calls == 1


def test_parity_composition_closes_baseline_when_candidate_creation_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    baseline = RecordingProvider()
    factory_calls = 0

    def provider_factory(_config: object) -> RecordingProvider:
        nonlocal factory_calls
        factory_calls += 1
        if factory_calls == 1:
            return baseline
        raise RuntimeError("candidate construction failed")

    monkeypatch.setenv("TA_SDK_OPENAI_URL", "https://embedding.example.test/v1/embeddings")
    monkeypatch.setattr(compatibility_module, "create_embedding_provider", provider_factory)

    with pytest.raises(RuntimeError, match="candidate construction failed"):
        asyncio.run(
            compatibility_module._check_from_arguments(  # pyright: ignore[reportPrivateUsage]
                compatibility_module._parser().parse_args(["--mode", "parity"])  # pyright: ignore[reportPrivateUsage]
            )
        )

    assert baseline.close_calls == 1


def test_index_compatibility_uses_stored_vectors_without_a_baseline_provider(tmp_path: Path) -> None:
    vectors = _varied_vectors()
    candidate = StaticProvider(np.vstack((vectors[:2], vectors)))
    report = asyncio.run(
        run_index_compatibility_check(
            candidate,
            _artifacts(tmp_path, vectors),
            _corpus(),
            artifact_hashes={"manifest_sha256": "b" * 64, "faiss_sha256": "c" * 64, "documents_sha256": "d" * 64},
        )
    )

    assert report["mode"] == "index"
    assert report["reference"] == "verified_index"
    assert report["passed"] is True
    metric_checks = report["metric_checks"]
    metrics = report["metrics"]
    assert isinstance(metric_checks, dict)
    assert isinstance(metrics, dict)
    assert set(cast(dict[str, bool], metric_checks)) == set(INDEX_THRESHOLDS)
    assert "mean_top5_query_neighbor_overlap" not in metrics


def test_index_compatibility_rejects_transformed_candidate(tmp_path: Path) -> None:
    vectors = _varied_vectors()
    report = asyncio.run(
        run_index_compatibility_check(
            StaticProvider(np.vstack((vectors[:2], -vectors))),
            _artifacts(tmp_path, vectors),
            _corpus(),
            artifact_hashes={"manifest_sha256": "b" * 64, "faiss_sha256": "c" * 64, "documents_sha256": "d" * 64},
        )
    )

    assert report["passed"] is False
    metric_checks = report["metric_checks"]
    assert isinstance(metric_checks, dict)
    assert metric_checks["same_input_cosine_p5"] is False


def test_index_compatibility_rejects_missing_reference_vectors(tmp_path: Path) -> None:
    vectors = _vectors()
    artifacts = _artifacts(tmp_path, vectors)
    invalid_artifacts = LoadedArtifacts(object(), artifacts.documents, artifacts.manifest, artifacts.paths)

    with pytest.raises(CompatibilityError, match="reference vectors"):
        asyncio.run(
            run_index_compatibility_check(
                StaticProvider(np.vstack((vectors[:2], vectors))),
                invalid_artifacts,
                _corpus(),
            )
        )


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
