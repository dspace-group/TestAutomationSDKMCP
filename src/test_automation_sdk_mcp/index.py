"""FAISS and generated-artifact I/O with cross-file generation validation."""

import hmac
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss

from .chunking import hash_file
from .documents import DocumentStore, IndexManifest, load_document_store, load_index_manifest
from .errors import ArtifactError, IndexBuildError

FAISS_FILENAME = "TA_Docu.faiss"
DOCUMENTS_FILENAME = "TA_Docu.documents.json"
MANIFEST_FILENAME = "TA_Docu.manifest.json"


@dataclass(frozen=True, slots=True)
class ArtifactPaths:
    """The three files that make up one published index generation."""

    directory: Path

    @property
    def faiss(self) -> Path:
        return self.directory / FAISS_FILENAME

    @property
    def documents(self) -> Path:
        return self.directory / DOCUMENTS_FILENAME

    @property
    def manifest(self) -> Path:
        return self.directory / MANIFEST_FILENAME


@dataclass(frozen=True, slots=True)
class LoadedArtifacts:
    """Verified artifacts ready for read-only retrieval."""

    index: Any
    documents: DocumentStore
    manifest: IndexManifest
    paths: ArtifactPaths


def artifact_paths(directory: Path) -> ArtifactPaths:
    """Return conventional artifact paths for an output directory."""

    return ArtifactPaths(directory=directory)


def _json_bytes(value: DocumentStore | IndexManifest) -> bytes:
    payload = json.dumps(
        value.model_dump(mode="json"),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{payload}\n".encode()


def write_document_store(path: Path, store: DocumentStore) -> None:
    """Write a document store with stable UTF-8 JSON formatting."""

    try:
        path.write_bytes(_json_bytes(store))
    except OSError as error:
        raise ArtifactError("Unable to write the document artifact.") from error


def write_index_manifest(path: Path, manifest: IndexManifest) -> None:
    """Write an index manifest with stable UTF-8 JSON formatting."""

    try:
        path.write_bytes(_json_bytes(manifest))
    except OSError as error:
        raise ArtifactError("Unable to write the index manifest.") from error


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise ArtifactError(f"Published {description} artifact is missing.")


def _verify_hash(path: Path, expected: str, description: str) -> None:
    try:
        actual = hash_file(path)
    except IndexBuildError as error:
        raise ArtifactError(f"Published {description} artifact could not be hashed.") from error
    if not hmac.compare_digest(actual, expected):
        raise ArtifactError(f"Published {description} artifact hash does not match its manifest.")


def _read_faiss(path: Path) -> Any:
    try:
        return faiss.read_index(str(path))
    except (OSError, RuntimeError, TypeError) as error:
        raise ArtifactError("Published FAISS artifact could not be read.") from error


def validate_faiss_index(index: Any, manifest: IndexManifest, documents: DocumentStore) -> None:
    """Validate FAISS type, dimension, row count, and document-row mapping."""

    if not isinstance(index, faiss.IndexFlatL2):
        raise ArtifactError("Published FAISS artifact is not an IndexFlatL2 index.")
    if index.d != manifest.embedding_dimension:
        raise ArtifactError("FAISS dimension does not match the index manifest.")
    if index.ntotal != manifest.document_count or index.ntotal != len(documents.documents):
        raise ArtifactError("FAISS and document artifact counts do not match.")
    if index.metric_type != faiss.METRIC_L2:
        raise ArtifactError("Published FAISS artifact does not use L2 distance.")


def load_verified_artifacts(directory: Path) -> LoadedArtifacts:
    """Load one complete generation and reject mixed or partially published files."""

    paths = artifact_paths(directory)
    _require_file(paths.manifest, "manifest")
    _require_file(paths.documents, "documents")
    _require_file(paths.faiss, "FAISS")
    manifest = load_index_manifest(paths.manifest)
    documents = load_document_store(paths.documents)
    if manifest.document_count != len(documents.documents):
        raise ArtifactError("Index manifest and document store counts do not match.")
    _verify_hash(paths.documents, manifest.documents_sha256, "documents")
    _verify_hash(paths.faiss, manifest.faiss_sha256, "FAISS")
    index = _read_faiss(paths.faiss)
    validate_faiss_index(index, manifest, documents)
    return LoadedArtifacts(index=index, documents=documents, manifest=manifest, paths=paths)


def load_artifact_set(directory: Path) -> LoadedArtifacts:
    """Compatibility name for the verified runtime artifact loader."""

    return load_verified_artifacts(directory)


__all__ = [
    "DOCUMENTS_FILENAME",
    "FAISS_FILENAME",
    "MANIFEST_FILENAME",
    "ArtifactPaths",
    "LoadedArtifacts",
    "artifact_paths",
    "load_artifact_set",
    "load_verified_artifacts",
    "validate_faiss_index",
    "write_document_store",
    "write_index_manifest",
]
