import os
import random
from pathlib import Path
from PIL import Image, ImageEnhance, ImageOps
from tqdm import tqdm


TARGET_COUNT = 500


def random_transform(image):
    transforms = [
        lambda img: img.transpose(Image.FLIP_LEFT_RIGHT),
        lambda img: img.rotate(random.randint(-25, 25)),
        lambda img: ImageEnhance.Brightness(img).enhance(random.uniform(0.7, 1.3)),
        lambda img: ImageEnhance.Contrast(img).enhance(random.uniform(0.7, 1.3)),
        lambda img: ImageEnhance.Color(img).enhance(random.uniform(0.7, 1.3)),
        lambda img: ImageOps.equalize(img),
        lambda img: img.crop(
            (
                random.randint(0, 50),
                random.randint(0, 50),
                img.width - random.randint(0, 50),
                img.height - random.randint(0, 50),
            )
        ),
    ]

    num_transforms = random.randint(1, 3)
    selected = random.sample(transforms, num_transforms)

    result = image
    for t in selected:
        try:
            result = t(result)
        except Exception:
            pass

    return result


def augment_class(class_dir: Path, target_count: int):
    images = (
        list(class_dir.glob("*.jpg"))
        + list(class_dir.glob("*.jpeg"))
        + list(class_dir.glob("*.png"))
    )
    current_count = len(images)

    if current_count >= target_count:
        return 0

    num_to_generate = target_count - current_count
    print(
        f"  {class_dir.name}: {current_count} -> {target_count} (generating {num_to_generate} augmented images)"
    )

    for i in range(num_to_generate):
        source_img = random.choice(images)
        try:
            with Image.open(source_img) as img:
                img = img.convert("RGB")
                augmented = random_transform(img)

                output_path = class_dir / f"aug_{i:05d}.jpg"
                augmented.save(output_path, quality=85)
        except Exception as e:
            print(f"    Error augmenting {source_img}: {e}")

    return num_to_generate


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Augment minority classes")
    parser.add_argument(
        "--data", "-d", default="data/consolidated", help="Dataset path"
    )
    parser.add_argument(
        "--target", "-t", type=int, default=TARGET_COUNT, help="Target images per class"
    )
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"Dataset not found: {data_path}")
        return

    classes = [d for d in data_path.iterdir() if d.is_dir()]
    print(f"Found {len(classes)} classes in {data_path}")

    total_augmented = 0
    for class_dir in tqdm(sorted(classes), desc="Augmenting classes"):
        count = augment_class(class_dir, args.target)
        total_augmented += count

    print(f"\nTotal augmented images: {total_augmented}")

    final_counts = {}
    for class_dir in classes:
        final_counts[class_dir.name] = len(
            list(class_dir.glob("*.jpg"))
            + list(class_dir.glob("*.jpeg"))
            + list(class_dir.glob("*.png"))
        )

    print("\nFinal class distribution:")
    for name, count in sorted(final_counts.items(), key=lambda x: x[1]):
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
