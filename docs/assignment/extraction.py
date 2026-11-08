"""Extract gzip-compressed data files from the project's data directory."""

from __future__ import annotations

import argparse
import gzip
import shutil
from pathlib import Path


def extract_gzip_files(data_dir: Path, overwrite: bool = False) -> list[Path]:
    """Extract every .gz file in ``data_dir`` and return created file paths."""
    extracted_files: list[Path] = []

    compressed_files = sorted(
        file_path
        for file_path in data_dir.iterdir()
        if file_path.is_file() and file_path.suffix.lower() == ".gz"
    )

    for compressed_file in compressed_files:
        output_file = compressed_file.with_suffix("")

        if output_file.exists() and not overwrite:
            print(f"Skipped (already exists): {output_file.name}")
            continue

        with gzip.open(compressed_file, "rb") as source, output_file.open("wb") as destination:
            shutil.copyfileobj(source, destination)

        extracted_files.append(output_file)
        print(f"Extracted: {compressed_file.name} -> {output_file.name}")

    return extracted_files


def main() -> None:
    project_dir = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Extract .gz files from the data directory.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_dir / "data",
        help="Directory containing compressed data files (default: project data directory).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite extracted files that already exist.",
    )
    args = parser.parse_args()

    if not args.data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {args.data_dir}")

    extract_gzip_files(args.data_dir, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
