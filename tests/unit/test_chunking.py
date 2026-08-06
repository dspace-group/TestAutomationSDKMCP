import json
from pathlib import Path

import pytest

from test_automation_sdk_mcp.chunking import (
    SearchItem,
    chunk_items,
    hash_html_tree,
    load_source_items,
    normalize_html_fragment,
    split_body,
)
from test_automation_sdk_mcp.errors import IndexBuildError


def write_source(root: Path, items: list[dict[str, object]], pages: tuple[str, ...] = ("index.html",)) -> Path:
    source = root / "data"
    source.mkdir(parents=True)
    for page in pages:
        page_path = source / page
        page_path.parent.mkdir(parents=True, exist_ok=True)
        page_path.write_bytes(page.encode("utf-8"))
    (source / "404.html").write_text("ignored", encoding="utf-8")
    (source / "search.json").write_text(json.dumps({"config": {}, "items": items}), encoding="utf-8")
    return source


def search_item(location: str = "index.html#intro") -> dict[str, object]:
    return {
        "location": location,
        "level": 1,
        "title": "Intro",
        "text": "<p>Hello &amp; world</p>",
        "path": ["Introduction"],
        "tags": ["guide"],
    }


def test_source_coverage_accepts_complete_pages_and_ignores_404(tmp_path: Path) -> None:
    source = write_source(tmp_path, [search_item()], ("index.html", "guide.html"))
    items = [search_item(), {**search_item("guide.html"), "title": "Guide"}]
    (source / "search.json").write_text(json.dumps({"items": items}), encoding="utf-8")

    assert len(load_source_items(source)) == 2


@pytest.mark.parametrize(
    "items, pages",
    [
        ([search_item()], ("index.html", "missing.html")),
        ([search_item(), search_item("missing.html")], ("index.html",)),
        ([search_item("../index.html")], ("index.html",)),
        ([search_item("C:/index.html")], ("index.html",)),
    ],
)
def test_source_coverage_rejects_missing_extra_and_unsafe_locations(
    tmp_path: Path, items: list[dict[str, object]], pages: tuple[str, ...]
) -> None:
    source = write_source(tmp_path, items, pages)

    with pytest.raises(IndexBuildError):
        load_source_items(source)


def test_html_normalization_preserves_entities_lists_code_and_line_endings() -> None:
    normalized = normalize_html_fragment(
        "<p>Hello &amp; welcome</p>\r\n<ul><li>one</li><li>two</li></ul>\r\n"
        "<pre><code>def run():\r\n    return 1</code></pre>"
    )

    assert "Hello & welcome" in normalized
    assert "- one\n- two" in normalized
    assert "```\ndef run():\n    return 1\n```" in normalized
    assert "\r" not in normalized
    assert "\n\n\n" not in normalized


def test_split_body_has_exact_overlap_for_unbroken_text() -> None:
    body = "0123456789" * 220

    chunks = split_body(body)

    assert [len(chunk) for chunk in chunks] == [1000, 1000, 600]
    assert chunks[0][-200:] == chunks[1][:200]
    assert chunks[1][-200:] == chunks[2][:200]


def test_chunks_have_stable_ids_and_embedding_only_prefix(tmp_path: Path) -> None:
    item = SearchItem(
        location="index.html#intro",
        level=1,
        title="Intro",
        text="<p>Body</p>",
        breadcrumbs=("Introduction",),
        tags=("guide",),
    )

    first = chunk_items((item,))[0]
    second = chunk_items((item,))[0]

    assert first.id == second.id
    assert first.content == "Body"
    assert "Introduction > Intro" in first.embedding_text
    assert first.content in first.embedding_text
    assert first.breadcrumbs == ("Introduction",)
    assert first.tags == ("guide",)


def test_html_tree_hash_is_independent_of_creation_order(tmp_path: Path) -> None:
    first = write_source(tmp_path / "first", [search_item()], ("index.html", "guide/page.html"))
    second = write_source(tmp_path / "second", [search_item()], ("guide/page.html", "index.html"))

    assert hash_html_tree(first) == hash_html_tree(second)
