import argparse
import sys
from pathlib import Path
from typing import List, Tuple, Dict

import torch
import numpy as np
from PIL import Image

import config
from src import create_model, get_transforms, load_data


def load_model_for_inference(
    model_path: str, num_classes: int, model_name: str, device: str
):
    model = create_model(model_name, num_classes, pretrained=False, dropout=0.0).to(
        device
    )
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


def get_top_k_predictions(
    model, image_tensor: torch.Tensor, label_encoder, device: str, top_k: int = 5
) -> List[Tuple[str, float]]:
    model.eval()
    with torch.no_grad():
        image_tensor = image_tensor.to(device).unsqueeze(0)
        logits = model(image_tensor)
        probs = torch.softmax(logits, dim=1)[0]

        top_probs, top_indices = torch.topk(probs, min(top_k, len(probs)))

        results = []
        for prob, idx in zip(top_probs.cpu().numpy(), top_indices.cpu().numpy()):
            class_name = label_encoder.inverse_transform([idx])[0]
            results.append((class_name, float(prob)))

    return results


def infer(
    image_path: str,
    model_path: str = config.BEST_MODEL_PATH,
    data_path: str = config.DATA_PATH,
    top_k: int = config.TOP_K_PREDICTIONS,
    model_name: str = config.MODEL_NAME,
) -> List[Tuple[str, float]]:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not Path(image_path).exists():
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)

    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first by running: python main.py")
        sys.exit(1)

    if not Path(data_path).exists():
        print(f"Error: Data path not found at {data_path}")
        sys.exit(1)

    df, label_encoder = load_data(data_path)
    num_classes = len(label_encoder.classes_)

    model = load_model_for_inference(model_path, num_classes, model_name, device)
    test_transform = get_transforms(config.IMG_SIZE, augment=False)

    image = Image.open(image_path).convert("RGB")
    image_tensor = test_transform(image)

    predictions = get_top_k_predictions(
        model, image_tensor, label_encoder, device, top_k
    )

    return predictions


def main():
    parser = argparse.ArgumentParser(description="Animal Face Classifier Inference")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument(
        "--model",
        "-m",
        default=config.BEST_MODEL_PATH,
        help=f"Path to model file (default: {config.BEST_MODEL_PATH})",
    )
    parser.add_argument(
        "--data",
        "-d",
        default=config.DATA_PATH,
        help=f"Path to dataset (default: {config.DATA_PATH})",
    )
    parser.add_argument(
        "--top-k",
        "-k",
        type=int,
        default=config.TOP_K_PREDICTIONS,
        help=f"Number of top predictions to show (default: {config.TOP_K_PREDICTIONS})",
    )
    args = parser.parse_args()

    predictions = infer(args.image, args.model, args.data, args.top_k)

    print(f"\nImage: {args.image}")
    print(f"\nTop {len(predictions)} Predictions:")
    print("-" * 40)

    max_class_len = max(len(cls) for cls, _ in predictions)

    for i, (class_name, prob) in enumerate(predictions, 1):
        bar_len = int(prob * 30)
        bar = "█" * bar_len + "░" * (30 - bar_len)
        print(f"{i}. {class_name:<{max_class_len}} {bar} {prob:.1%}")

    print("-" * 40)
    print(f"\nPrediction: {predictions[0][0]} ({predictions[0][1]:.1%})")


if __name__ == "__main__":
    main()
