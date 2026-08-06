from __future__ import annotations

import argparse
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

PACKAGE_NAME = "test-automation-sdk-mcp"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create the release archive.")
    parser.add_argument("--tag", required=True, help="Release tag used in the archive filename")
    parser.add_argument("--dist", type=Path, default=Path("dist"), help="Distribution directory")
    parser.add_argument(
        "--license-file",
        type=Path,
        default=Path("THIRD_PARTY_LICENSES.txt"),
        help="Generated third-party license file",
    )
    return parser.parse_args()


def find_single_file(directory: Path, pattern: str, description: str) -> Path:
    files = sorted(directory.glob(pattern))
    if len(files) != 1:
        raise SystemExit(f"expected exactly one {description} in {directory}, found {files}")
    return files[0]


def create_release_archive(release_tag: str, distribution_directory: Path, license_file: Path) -> Path:
    if not release_tag or Path(release_tag).name != release_tag:
        raise SystemExit(f"release tag must be a simple filename component: {release_tag!r}")

    wheel = find_single_file(distribution_directory, "*.whl", "wheel")
    source_archive = find_single_file(distribution_directory, "*.tar.gz", "source archive")
    if not license_file.is_file():
        raise SystemExit(f"third-party license file does not exist: {license_file}")

    archive = distribution_directory / f"{PACKAGE_NAME}-{release_tag}.zip"
    expected_members = [wheel.name, source_archive.name, "license.txt"]
    with ZipFile(archive, "w", compression=ZIP_DEFLATED) as release_zip:
        release_zip.write(wheel, wheel.name)
        release_zip.write(source_archive, source_archive.name)
        release_zip.write(license_file, "license.txt")

    with ZipFile(archive) as release_zip:
        actual_members = release_zip.namelist()
    if actual_members != expected_members:
        raise SystemExit(f"release archive members {actual_members} do not match {expected_members}")

    print(f"created {archive}")
    return archive


def main() -> int:
    arguments = parse_arguments()
    create_release_archive(arguments.tag, arguments.dist, arguments.license_file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
