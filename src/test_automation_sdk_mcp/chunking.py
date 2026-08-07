"""Deterministic validation, normalization, and chunking of documentation."""

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import cast

from markdownify import markdownify

from .errors import IndexBuildError

DEFAULT_MAX_CHARACTERS = 1000
DEFAULT_OVERLAP_CHARACTERS = 200
SEARCH_JSON_NAME = "search.json"


@dataclass(frozen=True, slots=True)
class SearchItem:
    """One validated record from the generated search index."""

    location: str
    level: int
    title: str
    text: str
    breadcrumbs: tuple[str, ...]
    tags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DocumentChunk:
    """One normalized chunk and the text used to create its embedding."""

    id: str
    location: str
    title: str
    breadcrumbs: tuple[str, ...]
    tags: tuple[str, ...]
    chunk_index: int
    content: str
    embedding_text: str


def _required_string(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise IndexBuildError(f"Search index field {name!r} must be a string.")
    return value


def _string_tuple(value: object, name: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise IndexBuildError(f"Search index field {name!r} must be a list of strings.")
    result: list[str] = []
    for item in cast(list[object], value):
        if not isinstance(item, str):
            raise IndexBuildError(f"Search index field {name!r} must be a list of strings.")
        result.append(item.strip())
    return tuple(result)


def _parse_search_item(value: object) -> SearchItem:
    if not isinstance(value, Mapping):
        raise IndexBuildError("Search index items must be objects.")
    item = cast(Mapping[str, object], value)
    location = _required_string(item.get("location"), "location").strip()
    title = _required_string(item.get("title"), "title").strip()
    text = _required_string(item.get("text"), "text")
    level = item.get("level")
    if isinstance(level, bool) or not isinstance(level, int) or level < 1:
        raise IndexBuildError("Search index field 'level' must be a positive integer.")
    return SearchItem(
        location=location,
        level=level,
        title=title,
        text=text,
        breadcrumbs=_string_tuple(item.get("path"), "path"),
        tags=_string_tuple(item.get("tags"), "tags"),
    )


def load_search_items(path: Path) -> tuple[SearchItem, ...]:
    """Load and validate the canonical ``search.json`` structure."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise IndexBuildError("Unable to read a valid search index.") from error
    if not isinstance(payload, Mapping):
        raise IndexBuildError("Search index must contain an items list.")
    mapping = cast(Mapping[str, object], payload)
    items = mapping.get("items")
    if not isinstance(items, list):
        raise IndexBuildError("Search index must contain an items list.")
    return tuple(_parse_search_item(item) for item in cast(list[object], items))


def _split_location(location: str) -> tuple[str, str | None]:
    if not location or "\x00" in location or "\\" in location:
        raise IndexBuildError("Documentation locations must be safe relative POSIX paths.")
    page, separator, anchor = location.partition("#")
    if not page or separator and not anchor:
        raise IndexBuildError("Documentation locations must contain a page path.")
    path = PurePosixPath(page)
    if (
        path.is_absolute()
        or PureWindowsPath(page).drive
        or page.startswith("/")
        or "//" in page
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise IndexBuildError("Documentation locations must be safe relative POSIX paths.")
    return page, anchor or None


def location_page(location: str) -> str:
    """Return a validated location's page path without its optional anchor."""

    return _split_location(location)[0]


def _html_pages(source_directory: Path) -> tuple[str, ...]:
    try:
        source_root = source_directory.resolve(strict=True)
    except OSError as error:
        raise IndexBuildError("Documentation source directory is unavailable.") from error
    if not source_root.is_dir():
        raise IndexBuildError("Documentation source must be a directory.")
    pages: list[str] = []
    for path in source_root.rglob("*.html"):
        if not path.is_file():
            continue
        relative = path.relative_to(source_root).as_posix()
        if relative == "404.html":
            continue
        pages.append(relative)
    return tuple(sorted(pages))


def validate_source_coverage(source_directory: Path, items: Sequence[SearchItem]) -> None:
    """Require exact page coverage and validate every search location."""

    html_pages = set(_html_pages(source_directory))
    search_pages: set[str] = set()
    for item in items:
        page = location_page(item.location)
        search_pages.add(page)
        if page not in html_pages:
            raise IndexBuildError("Search index contains a location without a source HTML page.")
    if search_pages != html_pages:
        raise IndexBuildError("Search index and HTML source page coverage do not match.")


def load_source_items(source_directory: Path) -> tuple[SearchItem, ...]:
    """Load ``search.json`` and validate it against the raw HTML tree."""

    items = load_search_items(source_directory / SEARCH_JSON_NAME)
    validate_source_coverage(source_directory, items)
    return items


def normalize_html_fragment(fragment: str) -> str:
    """Convert an HTML section fragment to stable Markdown text."""

    normalized_input = fragment.replace("\r\n", "\n").replace("\r", "\n")
    try:
        markdown = markdownify(
            normalized_input,
            bullets="-",
            heading_style="ATX",
            strip=["img", "script", "style"],
        )
    except (TypeError, ValueError) as error:
        raise IndexBuildError("Documentation HTML could not be normalized.") from error
    markdown = markdown.replace("\r\n", "\n").replace("\r", "\n")
    lines = [line.rstrip() for line in markdown.split("\n")]
    normalized = "\n".join(lines)
    normalized = re.sub(r"\n[ \t]*\n(?:[ \t]*\n)+", "\n\n", normalized)
    return normalized.strip()


def _split_end(text: str, start: int, limit: int) -> int:
    segment = text[start:limit]
    boundary = re.compile(r"\n\n|\n|(?<=[.!?])[ \t]+|[ \t]+")
    matches = list(boundary.finditer(segment))
    if not matches:
        return limit
    candidate = start + matches[-1].start()
    if candidate <= start:
        return limit
    return candidate


def split_body(
    body: str,
    *,
    max_characters: int = DEFAULT_MAX_CHARACTERS,
    overlap_characters: int = DEFAULT_OVERLAP_CHARACTERS,
) -> tuple[str, ...]:
    """Split a body at stable textual boundaries with a fixed character overlap."""

    if max_characters <= 0 or overlap_characters < 0 or overlap_characters >= max_characters:
        raise ValueError("overlap_characters must be between zero and max_characters - 1")
    if len(body) <= max_characters:
        return (body,) if body else ()

    chunks: list[str] = []
    start = 0
    while start < len(body):
        limit = min(start + max_characters, len(body))
        end = limit if limit == len(body) else _split_end(body, start, limit)
        if end - start <= overlap_characters:
            end = limit
        chunk = body[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(body):
            break
        next_start = end - overlap_characters
        if next_start <= start:
            next_start = start + max_characters - overlap_characters
        start = next_start
    return tuple(chunks)


def _chunk_id(location: str, chunk_index: int, content: str) -> str:
    identity = f"{location}\0{chunk_index}\0{content}".encode()
    return hashlib.sha256(identity).hexdigest()


def embedding_text(breadcrumbs: Sequence[str], title: str, content: str) -> str:
    prefix_parts = [*breadcrumbs, title]
    prefix = " > ".join(part for part in prefix_parts if part)
    return f"{prefix}\n\n{content}" if prefix and content else prefix or content


def _embedding_text(item: SearchItem, content: str) -> str:
    return embedding_text(item.breadcrumbs, item.title, content)


def chunk_items(
    items: Sequence[SearchItem],
    *,
    max_characters: int = DEFAULT_MAX_CHARACTERS,
    overlap_characters: int = DEFAULT_OVERLAP_CHARACTERS,
) -> tuple[DocumentChunk, ...]:
    """Normalize and chunk search records while inheriting their metadata."""

    chunks: list[DocumentChunk] = []
    for item in items:
        content = normalize_html_fragment(item.text)
        if not item.title and not content:
            continue
        if not content:
            content = item.title
        for chunk_index, chunk_content in enumerate(
            split_body(
                content,
                max_characters=max_characters,
                overlap_characters=overlap_characters,
            )
        ):
            chunks.append(
                DocumentChunk(
                    id=_chunk_id(item.location, chunk_index, chunk_content),
                    location=item.location,
                    title=item.title,
                    breadcrumbs=item.breadcrumbs,
                    tags=item.tags,
                    chunk_index=chunk_index,
                    content=chunk_content,
                    embedding_text=_embedding_text(item, chunk_content),
                )
            )
    return tuple(chunks)


def hash_file(path: Path) -> str:
    """Return the SHA-256 digest of a file's raw bytes."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as file:
            for block in iter(lambda: file.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        raise IndexBuildError("Unable to hash a source file.") from error
    return digest.hexdigest()


def hash_html_tree(source_directory: Path) -> str:
    """Hash sorted relative POSIX HTML paths together with their raw bytes."""

    digest = hashlib.sha256()
    source_root = source_directory.resolve()
    for relative in _html_pages(source_root):
        relative_bytes = relative.encode("utf-8")
        try:
            file_bytes = (source_root / Path(*PurePosixPath(relative).parts)).read_bytes()
        except OSError as error:
            raise IndexBuildError("Unable to hash the HTML source tree.") from error
        digest.update(len(relative_bytes).to_bytes(8, "big"))
        digest.update(relative_bytes)
        digest.update(len(file_bytes).to_bytes(8, "big"))
        digest.update(file_bytes)
    return digest.hexdigest()


def hash_source_tree(source_directory: Path) -> str:
    """Alias for the deterministic HTML source-tree hash."""

    return hash_html_tree(source_directory)


__all__ = [
    "DEFAULT_MAX_CHARACTERS",
    "DEFAULT_OVERLAP_CHARACTERS",
    "SEARCH_JSON_NAME",
    "DocumentChunk",
    "SearchItem",
    "chunk_items",
    "embedding_text",
    "hash_file",
    "hash_html_tree",
    "hash_source_tree",
    "load_search_items",
    "load_source_items",
    "location_page",
    "normalize_html_fragment",
    "split_body",
    "validate_source_coverage",
]
