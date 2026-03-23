import os
from typing import List, Tuple

import pandas as pd
import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def load_data(data_path: str) -> Tuple[pd.DataFrame, LabelEncoder]:
    image_paths: List[str] = []
    labels: List[str] = []

    first_level = os.listdir(data_path)

    if len(first_level) == 1 and os.path.isdir(os.path.join(data_path, first_level[0])):
        data_path = os.path.join(data_path, first_level[0])
        first_level = os.listdir(data_path)

    has_splits = any(
        s in ["train", "val", "test", "training", "validation"] for s in first_level
    )

    if has_splits:
        for split_dir in os.listdir(data_path):
            split_path = os.path.join(data_path, split_dir)
            if not os.path.isdir(split_path):
                continue
            for label in os.listdir(split_path):
                label_path = os.path.join(split_path, label)
                if not os.path.isdir(label_path):
                    continue
                for image in os.listdir(label_path):
                    if image.lower().endswith(
                        (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")
                    ):
                        image_paths.append(os.path.join(label_path, image))
                        labels.append(label)
    else:
        for label in os.listdir(data_path):
            label_path = os.path.join(data_path, label)
            if not os.path.isdir(label_path):
                continue
            for image in os.listdir(label_path):
                if image.lower().endswith(
                    (".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp")
                ):
                    image_paths.append(os.path.join(label_path, image))
                    labels.append(label)

    df = pd.DataFrame({"image_path": image_paths, "labels": labels})
    label_encoder = LabelEncoder()
    label_encoder.fit(df["labels"])
    return df, label_encoder


def split_data(
    df: pd.DataFrame, train_split: float, val_split: float, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.model_selection import train_test_split

    train, temp = train_test_split(
        df,
        test_size=(1 - train_split),
        random_state=random_state,
        stratify=df["labels"],
    )
    val_ratio = val_split / (val_split + (1 - train_split))
    val, test = train_test_split(
        temp,
        test_size=(1 - val_ratio),
        random_state=random_state,
        stratify=temp["labels"],
    )
    return (
        train.reset_index(drop=True),
        val.reset_index(drop=True),
        test.reset_index(drop=True),
    )


def get_transforms(
    img_size: int, augment: bool = False, for_heavy_augment: bool = False
):
    base_transforms: List = [transforms.Resize((img_size, img_size))]

    if augment:
        base_transforms.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomRotation(20),
                transforms.ColorJitter(
                    brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1
                ),
            ]
        )

        if for_heavy_augment:
            base_transforms.extend(
                [
                    transforms.RandomAffine(
                        degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)
                    ),
                    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
                ]
            )

    base_transforms.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    return transforms.Compose(base_transforms)


class CustomImageDataset(Dataset):
    def __init__(
        self,
        dataframe: pd.DataFrame,
        transform=None,
        label_encoder: LabelEncoder = None,
    ):
        self.dataframe = dataframe
        self.transform = transform
        self.label_encoder = label_encoder
        self.labels = None
        if label_encoder is not None:
            self.labels = torch.tensor(label_encoder.transform(dataframe["labels"]))

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx]["image_path"]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label
