import hashlib
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from PIL import Image
import random

ITALIAN_TO_ENGLISH = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "ragno": "spider",
    "scoiattolo": "squirrel",
}

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".webp"}
MIN_AUGMENTED_IMAGES = 500


def get_file_hash(filepath):
    hash_md5 = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()
    except Exception:
        return None


def is_valid_image(filepath):
    try:
        with Image.open(filepath) as img:
            img.verify()
        with Image.open(filepath) as img:
            img.load()
        return True
    except Exception:
        return False


def scan_dataset1(base_path):
    classes = {}
    dataset_path = Path(base_path) / "animals" / "animals"

    if not dataset_path.exists():
        print(f"Dataset1 path not found: {dataset_path}")
        return classes

    for class_name in dataset_path.iterdir():
        if class_name.is_dir():
            images = []
            for img_file in class_name.iterdir():
                if img_file.suffix.lower() in SUPPORTED_EXTENSIONS:
                    images.append(img_file)
            classes[class_name.name] = images

    return classes


def scan_dataset2(base_path):
    classes = {}
    dataset_path = Path(base_path) / "raw-img"

    if not dataset_path.exists():
        print(f"Dataset2 path not found: {dataset_path}")
        return classes

    for class_name in dataset_path.iterdir():
        if class_name.is_dir():
            english_name = ITALIAN_TO_ENGLISH.get(class_name.name, class_name.name)
            images = []
            for img_file in class_name.iterdir():
                if img_file.suffix.lower() in SUPPORTED_EXTENSIONS:
                    images.append(img_file)
            classes[english_name] = images

    return classes


def scan_dataset3(base_path):
    classes = {}
    dataset_path = Path(base_path) / "animals"

    if not dataset_path.exists():
        print(f"Dataset3 path not found: {dataset_path}")
        return classes

    for split in ["train", "val"]:
        split_path = dataset_path / split
        if not split_path.exists():
            continue
        for class_name in split_path.iterdir():
            if class_name.is_dir():
                if class_name.name not in classes:
                    classes[class_name.name] = []
                for img_file in class_name.iterdir():
                    if img_file.suffix.lower() in SUPPORTED_EXTENSIONS:
                        classes[class_name.name].append(img_file)

    return classes


def consolidate_datasets(output_path, min_images=100):
    output_path = Path(output_path)

    print("Scanning datasets...")

    all_classes = defaultdict(list)

    ds1 = scan_dataset1("data/new_dataset1")
    for cls, imgs in ds1.items():
        all_classes[cls].extend(imgs)

    ds2 = scan_dataset2("data/new_dataset2")
    for cls, imgs in ds2.items():
        all_classes[cls].extend(imgs)

    ds3 = scan_dataset3("data/new_dataset3")
    for cls, imgs in ds3.items():
        all_classes[cls].extend(imgs)

    print(f"\nFound {len(all_classes)} unique classes")

    seen_hashes = {}
    duplicates = 0
    corrupt = 0
    valid_images = defaultdict(list)

    print("\nValidating and deduplicating images...")
    for class_name, images in all_classes.items():
        for img_path in images:
            img_hash = get_file_hash(img_path)

            if img_hash is None:
                corrupt += 1
                continue

            if img_hash in seen_hashes:
                duplicates += 1
                continue

            if is_valid_image(img_path):
                seen_hashes[img_hash] = class_name
                valid_images[class_name].append(img_path)
            else:
                corrupt += 1

    print(f"  Duplicates removed: {duplicates}")
    print(f"  Corrupt images skipped: {corrupt}")

    print(f"\nClass distribution after cleaning:")
    class_counts = {cls: len(imgs) for cls, imgs in valid_images.items()}
    for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True)[
        :10
    ]:
        print(f"  {cls}: {count}")
    print(f"  ... and {len(class_counts) - 10} more classes")

    if output_path.exists():
        print(f"\nRemoving existing output directory: {output_path}")
        shutil.rmtree(output_path)

    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nCopying images to {output_path}...")
    for class_name, images in valid_images.items():
        class_dir = output_path / class_name
        class_dir.mkdir(exist_ok=True)
        for i, img_path in enumerate(images):
            dest = class_dir / f"{class_name}_{i:05d}{img_path.suffix.lower()}"
            shutil.copy2(img_path, dest)

    print(f"\nConsolidated dataset created at: {output_path}")
    print(f"Total classes: {len(valid_images)}")
    print(f"Total images: {sum(len(v) for v in valid_images.values())}")

    return output_path, class_counts


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Consolidate animal datasets")
    parser.add_argument(
        "--output", "-o", default="data/consolidated", help="Output path"
    )
    parser.add_argument(
        "--min-images", "-m", type=int, default=100, help="Minimum images per class"
    )
    args = parser.parse_args()

    consolidate_datasets(args.output, args.min_images)
