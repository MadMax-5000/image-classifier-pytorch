import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image
from sklearn.preprocessing import LabelEncoder

from src import CustomImageDataset, get_transforms, load_data, split_data


class TestDataLoading:
    def test_load_data_from_temp_dir(self, tmp_path):
        for label in ["cat", "dog", "wild"]:
            label_dir = tmp_path / label
            label_dir.mkdir()
            for i in range(3):
                img = Image.new("RGB", (64, 64), color=(i * 50, i * 50, i * 50))
                img.save(label_dir / f"img_{i}.jpg")

        df, label_encoder = load_data(str(tmp_path))

        assert len(df) == 9
        assert "image_path" in df.columns
        assert "labels" in df.columns
        assert set(df["labels"].unique()) == {"cat", "dog", "wild"}

    def test_load_data_handles_empty_dir(self, tmp_path):
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        df, label_encoder = load_data(str(empty_dir))

        assert len(df) == 0

    def test_load_data_skips_non_directories(self, tmp_path):
        test_file = tmp_path / "not_a_dir.txt"
        test_file.write_text("test")

        df, label_encoder = load_data(str(tmp_path))

        assert len(df) == 0


class TestDataSplit:
    def test_split_data_proportions(self):
        df = pd.DataFrame(
            {
                "image_path": [f"img_{i}.jpg" for i in range(100)],
                "labels": ["cat"] * 100,
            }
        )

        train, val, test = split_data(df, 0.7, 0.15)

        assert len(train) == 70
        assert len(val) == 15
        assert len(test) == 15
        assert len(train) + len(val) + len(test) == 100

    def test_split_data_no_overlap(self):
        df = pd.DataFrame(
            {
                "image_path": [f"img_{i}.jpg" for i in range(100)],
                "labels": ["cat"] * 100,
            }
        )

        train, val, test = split_data(df, 0.7, 0.15)

        train_paths = set(train["image_path"])
        val_paths = set(val["image_path"])
        test_paths = set(test["image_path"])

        assert len(train_paths & val_paths) == 0
        assert len(train_paths & test_paths) == 0
        assert len(val_paths & test_paths) == 0

    def test_split_data_deterministic(self):
        df = pd.DataFrame(
            {
                "image_path": [f"img_{i}.jpg" for i in range(100)],
                "labels": ["cat"] * 100,
            }
        )

        train1, val1, test1 = split_data(df, 0.7, 0.15)
        train2, val2, test2 = split_data(df, 0.7, 0.15)

        assert list(train1["image_path"]) == list(train2["image_path"])


class TestTransforms:
    def test_transforms_basic(self):
        transform = get_transforms(128, augment=False)
        assert transform is not None

        img = Image.new("RGB", (256, 256), color=(100, 100, 100))
        tensor = transform(img)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 128, 128)

    def test_transforms_with_augmentation(self):
        transform = get_transforms(128, augment=True)

        img = Image.new("RGB", (256, 256), color=(100, 100, 100))
        tensor = transform(img)

        assert isinstance(tensor, torch.Tensor)
        assert tensor.shape == (3, 128, 128)

    def test_transforms_different_sizes(self):
        for size in [64, 128, 256, 512]:
            transform = get_transforms(size, augment=False)
            img = Image.new("RGB", (1024, 1024), color=(100, 100, 100))
            tensor = transform(img)
            assert tensor.shape == (3, size, size)


class TestCustomDataset:
    @pytest.fixture
    def temp_dataset(self, tmp_path):
        for label in ["cat", "dog"]:
            label_dir = tmp_path / label
            label_dir.mkdir()
            for i in range(4):
                img = Image.new("RGB", (64, 64), color=(i * 50, i * 50, i * 50))
                img.save(label_dir / f"img_{i}.jpg")
        return tmp_path

    def test_dataset_length(self, temp_dataset):
        df, label_encoder = load_data(str(temp_dataset))
        transform = get_transforms(128)
        dataset = CustomImageDataset(df, transform, label_encoder)

        assert len(dataset) == 8

    def test_dataset_getitem(self, temp_dataset):
        df, label_encoder = load_data(str(temp_dataset))
        transform = get_transforms(128)
        dataset = CustomImageDataset(df, transform, label_encoder)

        image, label = dataset[0]

        assert isinstance(image, torch.Tensor)
        assert image.shape == (3, 128, 128)
        assert isinstance(label, torch.Tensor)
        assert label.item() in [0, 1]

    def test_dataset_without_transform(self, temp_dataset):
        df, label_encoder = load_data(str(temp_dataset))
        dataset = CustomImageDataset(df, None, label_encoder)

        image, label = dataset[0]

        assert isinstance(image, Image.Image)

    def test_dataset_iteration(self, temp_dataset):
        df, label_encoder = load_data(str(temp_dataset))
        transform = get_transforms(128)
        dataset = CustomImageDataset(df, transform, label_encoder)

        count = 0
        for img, label in dataset:
            count += 1
            assert isinstance(img, torch.Tensor)

        assert count == len(dataset)
