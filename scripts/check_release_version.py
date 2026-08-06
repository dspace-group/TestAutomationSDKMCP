from __future__ import annotations

import argparse
import tomllib
from pathlib import Path
from typing import Any, cast

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROJECT_FILE = REPOSITORY_ROOT / "pyproject.toml"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check a release tag against the package version.")
    parser.add_argument("release_tag", help="Release tag to validate, for example v0.1.0")
    parser.add_argument(
        "--project-file",
        type=Path,
        default=PROJECT_FILE,
        help="Path to pyproject.toml (defaults to the repository project file)",
    )
    return parser.parse_args()


def read_package_version(project_file: Path) -> str:
    with project_file.open("rb") as file_handle:
        project_data: dict[str, Any] = tomllib.load(file_handle)

    project_section_value = project_data.get("project")
    if not isinstance(project_section_value, dict):
        raise TypeError(f"project metadata is missing from {project_file}")
    project_section = cast(dict[str, object], project_section_value)

    package_version = project_section.get("version")
    if not isinstance(package_version, str) or not package_version:
        raise ValueError(f"project version is missing from {project_file}")
    return package_version


def main() -> int:
    arguments = parse_arguments()
    release_tag = arguments.release_tag
    if not release_tag.startswith("v"):
        raise SystemExit(f"release tag must start with 'v': {release_tag}")

    package_version = read_package_version(arguments.project_file)
    tag_version = release_tag.removeprefix("v")
    if tag_version != package_version:
        raise SystemExit(f"release tag version {tag_version!r} does not match package version {package_version!r}")

    print(f"package version {package_version} matches release tag {release_tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
