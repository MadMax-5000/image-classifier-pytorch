# Download Animal Faces HQ (AFHQ) Dataset
# Source: https://www.kaggle.com/datasets/andrewmvd/animal-faces

import argparse
import os
import shutil
import zipfile
from pathlib import Path

try:
    import kagglehub
except ImportError:
    print("kagglehub not installed. Install with: pip install kagglehub")
    exit(1)


def download_dataset(output_dir: str = "data"):
    print("Downloading Animal Faces HQ dataset...")

    path = kagglehub.dataset_download("andrewmvd/animal-faces")

    output_path = Path(output_dir) / "animal-faces"
    if output_path.exists():
        print(f"Dataset already exists at {output_path}")
        return str(output_path)

    os.makedirs(output_dir, exist_ok=True)

    source_path = Path(path)
    for item in source_path.iterdir():
        if item.is_dir():
            dest = output_path / item.name
            shutil.copytree(item, dest, dirs_exist_ok=True)
        else:
            dest = output_path / item.name
            shutil.copy2(item, dest)

    print(f"Dataset downloaded to: {output_path}")
    return str(output_path)


def verify_dataset(data_path: str):
    data_path = Path(data_path)
    expected_classes = {"cat", "dog", "wild"}

    found_classes = set()
    for folder in data_path.iterdir():
        if folder.is_dir():
            found_classes.add(folder.name)

    missing = expected_classes - found_classes
    if missing:
        print(f"Warning: Missing classes: {missing}")
        return False

    image_count = sum(1 for _ in data_path.rglob("*.jpg"))
    print(
        f"Dataset verified: {image_count} images found in {len(found_classes)} classes"
    )
    return True


def main():
    parser = argparse.ArgumentParser(description="Download AFHQ dataset")
    parser.add_argument(
        "--output", "-o", default="data", help="Output directory (default: data)"
    )
    parser.add_argument(
        "--verify", "-v", action="store_true", help="Verify dataset after download"
    )
    args = parser.parse_args()

    data_path = download_dataset(args.output)

    if args.verify:
        verify_dataset(data_path)


if __name__ == "__main__":
    main()
