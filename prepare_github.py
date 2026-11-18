"""Create a source-only upload bundle while preserving local development assets."""

import argparse
import shutil
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT_FILES = (
    ".gitignore",
    "README.md",
    "pyproject.toml",
    "requirements.txt",
    "requirements-ml.txt",
    "requirements-dev.txt",
    "config.example.json",
    "attack.example.json",
    "label_corrections.json",
    "launch.bat",
)


def source_files(root):
    files = [root / name for name in ROOT_FILES]
    files.extend(sorted(root.glob("*.py")))
    files.extend(sorted((root / "templates").rglob("*.png")))
    missing = [str(path.relative_to(root)) for path in files if not path.is_file()]
    if missing:
        raise ValueError("Required source files missing: " + ", ".join(missing))
    return files


def prepare(root, output):
    root = Path(root).resolve()
    output = Path(output).resolve()
    if root not in output.parents:
        raise ValueError("The upload folder must be inside the project directory")
    archive = output.with_suffix(".zip")
    if output.exists() or archive.exists():
        raise ValueError("Output already exists; choose a new --output to preserve it")
    files = source_files(root)
    output.mkdir(parents=True)
    for source in files:
        destination = output / source.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
    with ZipFile(archive, "w", ZIP_DEFLATED) as bundle:
        for source in files:
            relative = source.relative_to(root)
            bundle.write(output / relative, relative.as_posix())
    return archive, len(files)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="dist/github")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    archive, count = prepare(root, root / args.output)
    print(f"Prepared {count} source files in {archive.with_suffix('')}")
    print(f"ZIP: {archive} ({archive.stat().st_size:,} bytes)")
    print("No files were uploaded. Local datasets, recordings, and models were preserved.")


if __name__ == "__main__":
    main()
