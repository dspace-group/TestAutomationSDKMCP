"""Strict Pydantic models and JSON loaders for generated documentation artifacts.

Only top-level artifact models serialize ``schema_version``. Nested
``ChunkingManifest`` and ``DocumentRecord`` values are governed by the version
of their enclosing ``IndexManifest`` or ``DocumentStore``.
"""

import json
from pathlib import Path
from typing import Literal, Self, TypeGuard, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from .errors import ArtifactError

DOCUMENT_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 2
EMBEDDING_DIMENSION = 768
SHA256_LENGTH = 64


def _is_object_list(value: object) -> TypeGuard[list[object]]:
    return isinstance(value, list)


class _BoundaryModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ChunkingManifest(_BoundaryModel):
    """Deterministic chunking settings stored in a versioned index manifest."""

    max_characters: int = Field(gt=0)
    overlap_characters: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_overlap(self) -> Self:
        if self.overlap_characters >= self.max_characters:
            raise ValueError("overlap_characters must be less than max_characters")
        return self


class DocumentRecord(_BoundaryModel):
    """One ordered, embedded documentation chunk in a versioned document store."""

    id: str
    location: str
    title: str
    breadcrumbs: tuple[str, ...]
    tags: tuple[str, ...]
    chunk_index: int = Field(ge=0)
    content: str

    @field_validator("breadcrumbs", "tags", mode="before")
    @classmethod
    def normalize_json_arrays(cls, value: object) -> object:
        if _is_object_list(value):
            return tuple(value)
        return value

    @field_validator("id", "location", "content")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("value must not be empty")
        return value


class DocumentStore(_BoundaryModel):
    """Versioned document metadata whose array order maps to FAISS row IDs."""

    schema_version: Literal[1] = DOCUMENT_SCHEMA_VERSION
    documents: tuple[DocumentRecord, ...]

    @field_validator("documents", mode="before")
    @classmethod
    def normalize_json_array(cls, value: object) -> object:
        if _is_object_list(value):
            return tuple(value)
        return value

    @model_validator(mode="after")
    def validate_unique_ids(self) -> Self:
        document_ids = [document.id for document in self.documents]
        if len(document_ids) != len(set(document_ids)):
            raise ValueError("document IDs must be unique")
        return self


class IndexManifest(_BoundaryModel):
    """Versioned metadata describing one generated embedding index."""

    schema_version: Literal[2] = MANIFEST_SCHEMA_VERSION
    index_type: Literal["IndexFlatL2"]
    distance_metric: Literal["l2"]
    embedding_provider: Literal["ollama", "openai"]
    embedding_model: str | None
    embedding_dimension: int = Field(ge=1)
    document_count: int = Field(ge=0)
    search_json_sha256: str
    html_tree_sha256: str
    faiss_sha256: str
    documents_sha256: str
    chunking: ChunkingManifest

    @field_validator("embedding_model")
    @classmethod
    def validate_model_name(cls, value: str | None) -> str | None:
        if value is None:
            return value
        if not value.strip() or any(ord(char) < 32 for char in value):
            raise ValueError("embedding_model must be a non-empty model name")
        return value

    @model_validator(mode="after")
    def validate_provider_model(self) -> Self:
        if self.embedding_provider == "ollama" and self.embedding_model is None:
            raise ValueError("Ollama artifacts must record an embedding model")
        return self

    @field_validator("embedding_dimension")
    @classmethod
    def validate_embedding_dimension(cls, value: int) -> int:
        if value != EMBEDDING_DIMENSION:
            raise ValueError(f"embedding_dimension must be {EMBEDDING_DIMENSION}")
        return value

    @field_validator("search_json_sha256", "html_tree_sha256", "faiss_sha256", "documents_sha256")
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        if len(value) != SHA256_LENGTH or any(character not in "0123456789abcdef" for character in value):
            raise ValueError("artifact hashes must be lowercase SHA-256 hex strings")
        return value


def _load_json(path: Path) -> object:
    try:
        raw_json = path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ArtifactError("Unable to read documentation artifact.") from error
    try:
        return json.loads(raw_json)
    except json.JSONDecodeError as error:
        raise ArtifactError("Documentation artifact is not valid JSON.") from error


def _load_model(path: Path, model_type: type[DocumentStore] | type[IndexManifest]) -> DocumentStore | IndexManifest:
    payload = _load_json(path)
    if model_type is IndexManifest and isinstance(payload, dict):
        manifest_payload = cast(dict[str, object], payload)
        if manifest_payload.get("schema_version") != MANIFEST_SCHEMA_VERSION:
            raise ArtifactError(
                "Index manifest schema is obsolete; rebuild the index with this version of the package."
            )
    try:
        return model_type.model_validate(payload)
    except ValidationError as error:
        raise ArtifactError("Documentation artifact failed schema validation.") from error


def load_document_store(path: Path) -> DocumentStore:
    """Read and validate a document-store JSON artifact exactly once."""

    model = _load_model(path, DocumentStore)
    if not isinstance(model, DocumentStore):
        raise ArtifactError("Unexpected document-store artifact type.")
    return model


def load_index_manifest(path: Path) -> IndexManifest:
    """Read and validate an index-manifest JSON artifact exactly once."""

    model = _load_model(path, IndexManifest)
    if not isinstance(model, IndexManifest):
        raise ArtifactError("Unexpected index-manifest artifact type.")
    return model


def validate_artifact_consistency(manifest: IndexManifest, store: DocumentStore) -> None:
    """Validate the cross-file count contract before serving an index."""

    if manifest.document_count != len(store.documents):
        raise ArtifactError("Index manifest and document store counts do not match.")


def load_artifact_set(manifest_path: Path, documents_path: Path) -> tuple[IndexManifest, DocumentStore]:
    """Load both JSON artifacts and validate their shared document count."""

    manifest = load_index_manifest(manifest_path)
    store = load_document_store(documents_path)
    validate_artifact_consistency(manifest, store)
    return manifest, store
