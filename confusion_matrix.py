import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List
from PIL import Image

import config
from src import CustomImageDataset, Net, get_transforms, load_data


def get_predictions(model, loader, device: str = "cpu"):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    return np.array(all_preds), np.array(all_labels)


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    save_path: str = "confusion_matrix.png",
    normalize: bool = False,
):
    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        fmt = ".2%"
        title = "Normalized Confusion Matrix"
    else:
        fmt = "d"
        title = "Confusion Matrix"

    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        cbar_kws={"label": "Count"},
    )
    plt.title(title)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.show()
    print(f"Confusion matrix saved to {save_path}")


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names: List[str]):
    print("\n" + "=" * 50)
    print("Classification Report")
    print("=" * 50)
    print(classification_report(y_true, y_pred, target_names=class_names))

    print("\n" + "=" * 50)
    print("Summary Metrics")
    print("=" * 50)
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Macro F1-Score: {f1_score(y_true, y_pred, average='macro'):.4f}")
    print(f"Weighted F1-Score: {f1_score(y_true, y_pred, average='weighted'):.4f}")


def generate_confusion_matrix(
    model_path: str = config.MODEL_PATH,
    data_path: str = config.DATA_PATH,
    device: str = None,
    save_path: str = "confusion_matrix.png",
    normalize: bool = False,
):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    df, label_encoder = load_data(data_path)
    test_transform = get_transforms(config.IMG_SIZE, augment=False)

    train_df, val_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df["labels"]
    )
    test_df, _ = train_test_split(
        val_df, test_size=0.5, random_state=42, stratify=val_df["labels"]
    )

    test_dataset = CustomImageDataset(
        test_df.reset_index(drop=True), test_transform, label_encoder
    )
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    num_classes = len(label_encoder.classes_)
    model = Net(num_classes, config.DROPOUT).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))

    y_pred, y_true = get_predictions(model, test_loader, device)

    class_names = list(label_encoder.classes_)

    plot_confusion_matrix(y_true, y_pred, class_names, save_path, normalize)
    compute_metrics(y_true, y_pred, class_names)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate Confusion Matrix")
    parser.add_argument("--model", default=config.MODEL_PATH, help="Path to model")
    parser.add_argument("--data", default=config.DATA_PATH, help="Path to data")
    parser.add_argument("--output", default="confusion_matrix.png", help="Output path")
    parser.add_argument("--normalize", action="store_true", help="Normalize matrix")
    args = parser.parse_args()

    generate_confusion_matrix(
        args.model, args.data, save_path=args.output, normalize=args.normalize
    )
