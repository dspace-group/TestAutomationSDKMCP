import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from test_automation_sdk_mcp.documents import (
    ChunkingManifest,
    DocumentRecord,
    DocumentStore,
    IndexManifest,
    load_artifact_set,
    load_document_store,
    load_index_manifest,
    validate_artifact_consistency,
)
from test_automation_sdk_mcp.errors import ArtifactError


def make_document(document_id: str = "document-1", chunk_index: int = 0) -> DocumentRecord:
    return DocumentRecord(
        id=document_id,
        location="concepts/example.html#section",
        title="Example",
        breadcrumbs=("Concepts",),
        tags=("example",),
        chunk_index=chunk_index,
        content="Example content.",
    )


def make_manifest(document_count: int = 1, embedding_dimension: int = 768) -> IndexManifest:
    digest = "a" * 64
    return IndexManifest(
        schema_version=1,
        index_type="IndexFlatL2",
        distance_metric="l2",
        embedding_provider="ollama",
        embedding_model="nomic-embed-text:v1.5",
        embedding_dimension=embedding_dimension,
        document_count=document_count,
        search_json_sha256=digest,
        html_tree_sha256=digest,
        faiss_sha256=digest,
        documents_sha256=digest,
        chunking=ChunkingManifest(max_characters=1000, overlap_characters=200),
    )


def test_valid_models_are_frozen_and_dump_to_json_shape() -> None:
    store = DocumentStore(schema_version=1, documents=(make_document(),))
    manifest = make_manifest()

    assert store.model_dump(mode="json")["documents"][0]["breadcrumbs"] == ["Concepts"]
    assert "schema_version" not in store.model_dump(mode="json")["documents"][0]
    assert manifest.model_dump(mode="json")["chunking"] == {
        "max_characters": 1000,
        "overlap_characters": 200,
    }
    assert "schema_version" not in manifest.model_dump(mode="json")["chunking"]
    with pytest.raises(ValidationError):
        store.documents = ()  # type: ignore[misc]


def test_frozen_models_reject_nested_mutation() -> None:
    document = make_document()
    store = DocumentStore(schema_version=1, documents=(document,))

    with pytest.raises(TypeError):
        document.breadcrumbs[0] = "Changed"  # type: ignore[index]
    with pytest.raises(TypeError):
        store.documents[0].tags[0] = "Changed"  # type: ignore[index]


@pytest.mark.parametrize(
    "factory",
    [
        lambda: DocumentStore(schema_version=1, documents=(make_document(), make_document())),
        lambda: make_document(chunk_index=-1),
        lambda: make_manifest(embedding_dimension=384),
        lambda: make_manifest(document_count=-1),
        lambda: ChunkingManifest(max_characters=100, overlap_characters=100),
    ],
)
def test_invalid_cross_field_values_are_rejected(factory: object) -> None:
    with pytest.raises((ValidationError, ValueError)):
        factory()  # type: ignore[operator]


@pytest.mark.parametrize(
    "payload",
    [
        {"schema_version": 2, "documents": []},
        {"schema_version": 1, "documents": [], "unexpected": True},
    ],
)
def test_document_store_rejects_unknown_and_wrong_schema_fields(payload: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        DocumentStore.model_validate(payload)


def test_index_manifest_rejects_unknown_fields_and_wrong_schema_version() -> None:
    payload = make_manifest().model_dump()
    payload["schema_version"] = 2
    with pytest.raises(ValidationError):
        IndexManifest.model_validate(payload)

    payload = make_manifest().model_dump()
    payload["unexpected"] = True
    with pytest.raises(ValidationError):
        IndexManifest.model_validate(payload)


def test_json_loaders_validate_once_and_translate_failures(tmp_path: Path) -> None:
    documents_path = tmp_path / "documents.json"
    manifest_path = tmp_path / "manifest.json"
    store = DocumentStore(schema_version=1, documents=(make_document(),))
    manifest = make_manifest()
    documents_path.write_text(json.dumps(store.model_dump(mode="json")), encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest.model_dump(mode="json")), encoding="utf-8")

    assert load_document_store(documents_path) == store
    assert load_index_manifest(manifest_path) == manifest
    assert load_artifact_set(manifest_path, documents_path) == (manifest, store)

    documents_path.write_text("{not-json", encoding="utf-8")
    with pytest.raises(ArtifactError) as raised:
        load_document_store(documents_path)
    assert isinstance(raised.value.__cause__, json.JSONDecodeError)


def test_count_mismatch_is_rejected_without_exposing_paths() -> None:
    manifest = make_manifest(document_count=2)
    store = DocumentStore(schema_version=1, documents=(make_document(),))

    with pytest.raises(ArtifactError) as raised:
        validate_artifact_consistency(manifest, store)

    assert "counts do not match" in str(raised.value)
