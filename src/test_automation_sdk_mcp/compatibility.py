"""Advisory comparison of the packaged index embedding space and a candidate endpoint."""

import argparse
import asyncio
import json
import os
import sys
from collections.abc import Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from hashlib import sha256
from importlib.metadata import version
from importlib.resources import as_file, files
from pathlib import Path
from typing import Literal, Protocol, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from .chunking import embedding_text
from .config import EmbeddingProviderKind, RuntimeConfig, runtime_config_from_environment
from .documents import EMBEDDING_DIMENSION
from .errors import CompatibilityError, TestAutomationSDKError
from .index import LoadedArtifacts, load_verified_artifacts, packaged_artifact_paths
from .provider import EmbeddingProvider
from .provider.factory import create_embedding_provider

PROBE_CORPUS_FILENAME = "embedding_compatibility_probe.json"
COMPATIBILITY_SCHEMA_VERSION = 1
DEFAULT_DOCUMENT_SAMPLE_SIZE = 128
DEFAULT_TOP_K = 10
THRESHOLDS: dict[str, float] = {
    "same_input_cosine_p5": 0.995,
    "pairwise_cosine_correlation": 0.995,
    "mean_top5_query_neighbor_overlap": 0.95,
    "mean_top10_document_neighborhood_overlap": 0.95,
}


class CompatibilityProbe(BaseModel):
    """One deterministic query and its representative result requirement."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    id: str
    text: str
    expected_locations: tuple[str, ...]

    @field_validator("expected_locations", mode="before")
    @classmethod
    def normalize_locations(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(cast(list[str], value))
        return value

    @field_validator("id", "text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("probe text fields must not be empty")
        return value


class ProbeCorpus(BaseModel):
    """Strict deterministic probe inputs shipped with the package."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1] = COMPATIBILITY_SCHEMA_VERSION
    queries: tuple[CompatibilityProbe, ...]
    document_row_ids: tuple[int, ...]

    @field_validator("queries", mode="before")
    @classmethod
    def normalize_queries(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(cast(list[CompatibilityProbe], value))
        return value

    @field_validator("document_row_ids", mode="before")
    @classmethod
    def normalize_document_rows(cls, value: object) -> object:
        if isinstance(value, list):
            return tuple(cast(list[int], value))
        return value

    @field_validator("queries")
    @classmethod
    def validate_queries(cls, value: tuple[CompatibilityProbe, ...]) -> tuple[CompatibilityProbe, ...]:
        if not value or len({probe.id for probe in value}) != len(value):
            raise ValueError("probe queries must be non-empty and uniquely named")
        return value

    @field_validator("document_row_ids")
    @classmethod
    def validate_rows(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value or len(value) > DEFAULT_DOCUMENT_SAMPLE_SIZE or len(set(value)) != len(value):
            raise ValueError("document probe rows must be unique and have a bounded sample size")
        if any(isinstance(row_id, bool) or row_id < 0 for row_id in value):
            raise ValueError("document probe rows must be non-negative integers")
        return value


@dataclass(frozen=True, slots=True)
class VectorInspection:
    """Safe structural facts and numeric values for one provider response."""

    values: NDArray[np.float64] | None
    dimension: int | None
    dtype: str
    finite: bool
    contiguous: bool
    shape_valid: bool
    norms: NDArray[np.float64] | None

    @property
    def structural_passed(self) -> bool:
        return (
            self.shape_valid
            and self.dtype == "float32"
            and self.finite
            and self.contiguous
            and self.norms is not None
            and bool(np.isfinite(self.norms).all())
            and bool((self.norms > 0).all())
        )


@dataclass(frozen=True, slots=True)
class _ComparisonValues:
    baseline_queries: NDArray[np.float64]
    candidate_queries: NDArray[np.float64]
    baseline_documents: NDArray[np.float64]
    candidate_documents: NDArray[np.float64]


def _invalid(message: str) -> CompatibilityError:
    return CompatibilityError(message)


def load_probe_corpus(path: Path) -> ProbeCorpus:
    """Load and validate one deterministic probe corpus."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise _invalid("Compatibility probe corpus is unavailable or invalid.") from error
    try:
        return ProbeCorpus.model_validate(payload)
    except ValidationError as error:
        raise _invalid("Compatibility probe corpus failed schema validation.") from error


@contextmanager
def packaged_probe_corpus() -> Generator[tuple[bytes, ProbeCorpus], None, None]:
    """Yield the packaged corpus bytes and validated model without exposing its path."""

    resource = files("test_automation_sdk_mcp").joinpath("db", PROBE_CORPUS_FILENAME)
    try:
        with as_file(resource) as path:
            raw = path.read_bytes()
            yield raw, ProbeCorpus.model_validate(json.loads(raw))
    except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError, ValidationError) as error:
        raise _invalid("Compatibility probe corpus is unavailable or invalid.") from error


def _inspect_vectors(value: object, expected_count: int) -> VectorInspection:
    try:
        raw = np.asarray(value)
    except (OverflowError, TypeError, ValueError):
        return VectorInspection(None, None, "invalid", False, False, False, None)

    shape_valid = raw.shape == (expected_count, EMBEDDING_DIMENSION)
    dimension = int(raw.shape[1]) if raw.ndim == 2 else None
    dtype = str(raw.dtype)
    contiguous = bool(raw.flags.c_contiguous)
    try:
        finite = bool(np.isfinite(raw).all())
    except (TypeError, ValueError):
        finite = False
    values: NDArray[np.float64] | None = None
    norms: NDArray[np.float64] | None = None
    if shape_valid and finite:
        try:
            values = np.asarray(raw, dtype=np.float64)
            norms = np.linalg.norm(values, axis=1)
        except (OverflowError, TypeError, ValueError):
            values = None
            norms = None
    return VectorInspection(values, dimension, dtype, finite, contiguous, shape_valid, norms)


def _stats(values: NDArray[np.float64] | None) -> dict[str, float | None]:
    if values is None or values.size == 0:
        return {"min": None, "mean": None, "max": None}
    return {
        "min": float(np.min(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def _cosine_values(left: NDArray[np.float64], right: NDArray[np.float64]) -> NDArray[np.float64]:
    left_norms = np.linalg.norm(left, axis=1)
    right_norms = np.linalg.norm(right, axis=1)
    return np.sum(left * right, axis=1) / (left_norms * right_norms)


def _pairwise_correlation(values: _ComparisonValues) -> float | None:
    combined_baseline = np.vstack((values.baseline_queries, values.baseline_documents))
    combined_candidate = np.vstack((values.candidate_queries, values.candidate_documents))
    baseline_norms = np.linalg.norm(combined_baseline, axis=1)
    candidate_norms = np.linalg.norm(combined_candidate, axis=1)
    baseline_normalized = combined_baseline / baseline_norms[:, None]
    candidate_normalized = combined_candidate / candidate_norms[:, None]
    baseline_geometry = baseline_normalized @ baseline_normalized.T
    candidate_geometry = candidate_normalized @ candidate_normalized.T
    upper = np.triu_indices_from(baseline_geometry, k=1)
    baseline_flat = baseline_geometry[upper]
    candidate_flat = candidate_geometry[upper]
    if baseline_flat.size < 2 or np.std(baseline_flat) == 0 or np.std(candidate_flat) == 0:
        return None
    correlation = float(np.corrcoef(baseline_flat, candidate_flat)[0, 1])
    return correlation if np.isfinite(correlation) else None


class _SearchIndex(Protocol):
    def search(self, query: NDArray[np.float32], top_k: int) -> tuple[NDArray[np.float32], NDArray[np.int64]]: ...


def _nearest_rows(index: _SearchIndex, values: NDArray[np.float64], top_k: int) -> tuple[tuple[int, ...], ...]:
    query = np.ascontiguousarray(values, dtype=np.float32)
    try:
        raw_distances, raw_ids = index.search(query, top_k)
        distances = np.asarray(raw_distances)
        ids = np.asarray(raw_ids)
    except (OverflowError, RuntimeError, TypeError, ValueError) as error:
        raise _invalid("Packaged FAISS index returned invalid compatibility results.") from error
    if distances.shape != ids.shape or distances.ndim != 2 or distances.shape[1] != top_k:
        raise _invalid("Packaged FAISS index returned invalid compatibility results.")
    if not np.isfinite(distances).all() or not np.issubdtype(ids.dtype, np.integer):
        raise _invalid("Packaged FAISS index returned invalid compatibility results.")
    return tuple(tuple(int(row_id) for row_id in row) for row in ids)


def _mean_overlap(left: Sequence[Sequence[int]], right: Sequence[Sequence[int]], top_k: int) -> float:
    if not left or len(left) != len(right):
        return 0.0
    return float(np.mean([len(set(a[:top_k]) & set(b[:top_k])) / top_k for a, b in zip(left, right, strict=True)]))


def _document_neighborhoods(values: NDArray[np.float64], row_ids: Sequence[int], top_k: int) -> tuple[set[int], ...]:
    norms = np.linalg.norm(values, axis=1)
    similarity = (values / norms[:, None]) @ (values / norms[:, None]).T
    neighborhoods: list[set[int]] = []
    for position, _row_id in enumerate(row_ids):
        ordered = np.argsort(-similarity[position], kind="stable")
        ordered_positions = cast(list[int], ordered.tolist())
        neighbors = [row_ids[index] for index in ordered_positions if index != position][:top_k]
        neighborhoods.append(set(neighbors))
    return tuple(neighborhoods)


def _neighborhood_overlap(baseline: Sequence[set[int]], candidate: Sequence[set[int]], top_k: int) -> float:
    if not baseline or len(baseline) != len(candidate):
        return 0.0
    return float(np.mean([len(left & right) / top_k for left, right in zip(baseline, candidate, strict=True)]))


def _metric_checks(metrics: dict[str, float | int | None | bool]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for name, threshold in THRESHOLDS.items():
        value = metrics.get(name)
        checks[name] = isinstance(value, (int, float)) and not isinstance(value, bool) and value >= threshold
    return checks


def _vector_quality(inspection: VectorInspection) -> dict[str, object]:
    return {
        "dimension": inspection.dimension,
        "dtype": inspection.dtype,
        "finite": inspection.finite,
        "contiguous": inspection.contiguous,
        "shape_valid": inspection.shape_valid,
        "positive_norms": inspection.norms is not None and bool((inspection.norms > 0).all()),
        "norms": _stats(inspection.norms),
        "passed": inspection.structural_passed,
    }


def _comparison_values(
    baseline: VectorInspection,
    candidate: VectorInspection,
    query_count: int,
    document_count: int,
) -> _ComparisonValues | None:
    if baseline.values is None or candidate.values is None:
        return None
    return _ComparisonValues(
        baseline.values[:query_count],
        candidate.values[:query_count],
        baseline.values[query_count : query_count + document_count],
        candidate.values[query_count : query_count + document_count],
    )


def _artifact_hashes(artifacts: LoadedArtifacts) -> dict[str, str]:
    return {
        "manifest_sha256": sha256(artifacts.paths.manifest.read_bytes()).hexdigest(),
        "faiss_sha256": artifacts.manifest.faiss_sha256,
        "documents_sha256": artifacts.manifest.documents_sha256,
    }


async def run_compatibility_check(
    baseline_provider: EmbeddingProvider,
    candidate_provider: EmbeddingProvider,
    artifacts: LoadedArtifacts,
    corpus: ProbeCorpus,
    *,
    corpus_sha256: str = "",
    artifact_hashes: dict[str, str] | None = None,
) -> dict[str, object]:
    """Compare two providers and return a safe, deterministic advisory report."""

    if any(row_id >= len(artifacts.documents.documents) for row_id in corpus.document_row_ids):
        raise _invalid("Compatibility probe corpus references a missing documentation row.")
    query_texts = [probe.text for probe in corpus.queries]
    document_texts = [
        embedding_text(
            artifacts.documents.documents[row_id].breadcrumbs,
            artifacts.documents.documents[row_id].title,
            artifacts.documents.documents[row_id].content,
        )
        for row_id in corpus.document_row_ids
    ]
    inputs = query_texts + document_texts
    baseline_raw = await baseline_provider.embed(inputs)
    candidate_raw = await candidate_provider.embed(inputs)
    baseline = _inspect_vectors(baseline_raw, len(inputs))
    candidate = _inspect_vectors(candidate_raw, len(inputs))
    values = _comparison_values(baseline, candidate, len(query_texts), len(document_texts))

    metrics: dict[str, float | int | None | bool] = {
        "same_input_cosine_min": None,
        "same_input_cosine_mean": None,
        "same_input_cosine_p5": None,
        "pairwise_cosine_correlation": None,
        "mean_top5_query_neighbor_overlap": None,
        "mean_top10_document_neighborhood_overlap": None,
        "candidate_self_retrieval_top1_rate": None,
        "candidate_self_retrieval_mrr10": None,
    }
    representative_results: dict[str, list[str]] = {}
    representative_passed = False
    metric_checks: dict[str, bool] = {name: False for name in THRESHOLDS}
    if values is not None and baseline.structural_passed and candidate.structural_passed:
        same_input = _cosine_values(values.baseline_queries, values.candidate_queries)
        same_document = _cosine_values(values.baseline_documents, values.candidate_documents)
        same_values = np.concatenate((same_input, same_document))
        metrics["same_input_cosine_min"] = float(np.min(same_values))
        metrics["same_input_cosine_mean"] = float(np.mean(same_values))
        metrics["same_input_cosine_p5"] = float(np.percentile(same_values, 5))
        metrics["pairwise_cosine_correlation"] = _pairwise_correlation(values)

        search_index = cast(_SearchIndex, artifacts.index)
        baseline_query_neighbors = _nearest_rows(search_index, values.baseline_queries, 5)
        candidate_query_neighbors = _nearest_rows(search_index, values.candidate_queries, 5)
        metrics["mean_top5_query_neighbor_overlap"] = _mean_overlap(
            baseline_query_neighbors, candidate_query_neighbors, 5
        )
        baseline_document_neighbors = _document_neighborhoods(
            values.baseline_documents, corpus.document_row_ids, DEFAULT_TOP_K
        )
        candidate_document_neighbors = _document_neighborhoods(
            values.candidate_documents, corpus.document_row_ids, DEFAULT_TOP_K
        )
        metrics["mean_top10_document_neighborhood_overlap"] = _neighborhood_overlap(
            baseline_document_neighbors, candidate_document_neighbors, DEFAULT_TOP_K
        )
        candidate_self_neighbors = _nearest_rows(search_index, values.candidate_documents, DEFAULT_TOP_K)
        reciprocal_ranks: list[float] = []
        top1_matches = 0
        for expected_row_id, neighbors in zip(corpus.document_row_ids, candidate_self_neighbors, strict=True):
            if neighbors and neighbors[0] == expected_row_id:
                top1_matches += 1
            try:
                reciprocal_ranks.append(1.0 / (neighbors.index(expected_row_id) + 1))
            except ValueError:
                reciprocal_ranks.append(0.0)
        metrics["candidate_self_retrieval_top1_rate"] = top1_matches / len(corpus.document_row_ids)
        metrics["candidate_self_retrieval_mrr10"] = float(np.mean(reciprocal_ranks))
        for probe, neighbors in zip(corpus.queries, candidate_query_neighbors, strict=True):
            locations = [artifacts.documents.documents[row_id].location for row_id in neighbors]
            representative_results[probe.id] = locations
        representative_probes = [probe for probe in corpus.queries if probe.expected_locations]
        representative_passed = all(
            any(
                any(location.startswith(prefix) for prefix in probe.expected_locations)
                for location in representative_results[probe.id]
            )
            for probe in representative_probes
        )
        metric_checks = _metric_checks(metrics)

    structural_passed = baseline.structural_passed and candidate.structural_passed
    passed = structural_passed and all(metric_checks.values()) and representative_passed
    return {
        "schema_version": COMPATIBILITY_SCHEMA_VERSION,
        "package_version": version("test-automation-sdk-mcp"),
        "artifact_hashes": _artifact_hashes(artifacts) if artifact_hashes is None else artifact_hashes,
        "corpus_sha256": corpus_sha256,
        "sampled_row_ids": list(corpus.document_row_ids),
        "thresholds": THRESHOLDS,
        "vector_quality": {
            "baseline": _vector_quality(baseline),
            "candidate": _vector_quality(candidate),
        },
        "metrics": metrics,
        "metric_checks": metric_checks,
        "representative_results": representative_results,
        "representative_passed": representative_passed,
        "structural_passed": structural_passed,
        "passed": passed,
    }


def _config_for(
    provider: EmbeddingProviderKind,
    *,
    endpoint_url: str | None,
    model: str | None,
) -> RuntimeConfig:
    values = dict(os.environ)
    values["TA_SDK_EMBEDDING_PROVIDER"] = provider.value
    if endpoint_url is not None:
        variable = "TA_SDK_OLLAMA_URL" if provider is EmbeddingProviderKind.OLLAMA else "TA_SDK_OPENAI_URL"
        values[variable] = endpoint_url
    if model is not None:
        variable = "TA_SDK_OLLAMA_MODEL" if provider is EmbeddingProviderKind.OLLAMA else "TA_SDK_OPENAI_MODEL"
        values[variable] = model
    return runtime_config_from_environment(values)


async def _check_from_arguments(args: argparse.Namespace) -> dict[str, object]:
    baseline_config = _config_for(
        EmbeddingProviderKind.OLLAMA,
        endpoint_url=args.ollama_url,
        model=args.ollama_model,
    )
    candidate_config = _config_for(
        EmbeddingProviderKind.OPENAI,
        endpoint_url=args.openai_url,
        model=args.openai_model,
    )
    baseline_provider = create_embedding_provider(baseline_config)
    candidate_provider = create_embedding_provider(candidate_config)
    try:
        with packaged_artifact_paths() as paths:
            artifacts = load_verified_artifacts(paths.directory)
            artifact_hashes = _artifact_hashes(artifacts)
            with packaged_probe_corpus() as loaded:
                raw_corpus, corpus = loaded
                return await run_compatibility_check(
                    baseline_provider,
                    candidate_provider,
                    artifacts,
                    corpus,
                    corpus_sha256=sha256(raw_corpus).hexdigest(),
                    artifact_hashes=artifact_hashes,
                )
    finally:
        for provider in (baseline_provider, candidate_provider):
            await provider.aclose()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare Ollama and OpenAI-compatible embedding behavior.")
    parser.add_argument("--ollama-url", help="Override the non-secret Ollama base URL.")
    parser.add_argument("--ollama-model", help="Override the non-secret Ollama model.")
    parser.add_argument("--openai-url", help="Override the non-secret OpenAI embeddings URL.")
    parser.add_argument("--openai-model", help="Override the non-secret OpenAI model.")
    parser.add_argument("--json", action="store_true", help="Emit the safe report as JSON.")
    return parser


def _human_summary(report: dict[str, object]) -> str:
    metrics = cast(dict[str, object], report["metrics"])
    status = "PASS" if report["passed"] else "ADVISORY FAIL"
    return "\n".join(
        [
            f"embedding compatibility: {status}",
            (
                f"same-input cosine p5={metrics['same_input_cosine_p5']} "
                f"pairwise-correlation={metrics['pairwise_cosine_correlation']}"
            ),
            (
                f"query top-5 overlap={metrics['mean_top5_query_neighbor_overlap']} "
                f"document top-10 overlap={metrics['mean_top10_document_neighborhood_overlap']}"
            ),
            (
                f"candidate self-retrieval top-1={metrics['candidate_self_retrieval_top1_rate']} "
                f"mrr@10={metrics['candidate_self_retrieval_mrr10']}"
            ),
        ]
    )


def main(arguments: Sequence[str] | None = None) -> int:
    """Run the advisory embedding compatibility command."""

    args = _parser().parse_args(arguments)
    try:
        report = asyncio.run(_check_from_arguments(args))
    except TestAutomationSDKError as error:
        print(f"error: {error.safe_message}", file=sys.stderr)
        return 1
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        print(f"error: compatibility check failed ({type(error).__name__})", file=sys.stderr)
        return 1
    if args.json:
        print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    else:
        print(_human_summary(report))
    return 0 if report["passed"] else 1


__all__ = [
    "COMPATIBILITY_SCHEMA_VERSION",
    "DEFAULT_DOCUMENT_SAMPLE_SIZE",
    "PROBE_CORPUS_FILENAME",
    "THRESHOLDS",
    "CompatibilityProbe",
    "ProbeCorpus",
    "load_probe_corpus",
    "main",
    "packaged_probe_corpus",
    "run_compatibility_check",
]
