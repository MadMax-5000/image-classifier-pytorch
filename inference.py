import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

import config
from src import CustomImageDataset, Net, get_transforms, load_data, predict


def load_model(model_path: str, num_classes: int, device: str):
    model = Net(num_classes, config.DROPOUT).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()
    return model


def infer(
    image_path: str,
    model_path: str = config.MODEL_PATH,
    data_path: str = config.DATA_PATH,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not Path(image_path).exists():
        print(f"Error: Image not found at {image_path}")
        sys.exit(1)

    if not Path(model_path).exists():
        print(f"Error: Model not found at {model_path}")
        print("Please train the model first by running: python main.py")
        sys.exit(1)

    df, label_encoder = load_data(data_path)
    num_classes = len(df["labels"].unique())

    model = load_model(model_path, num_classes, device)
    test_transform = get_transforms(config.IMG_SIZE, augment=False)

    image = Image.open(image_path).convert("RGB")
    image_tensor = test_transform(image)

    pred_idx = predict(model, image_tensor, device)
    prediction = label_encoder.inverse_transform([pred_idx])[0]

    print(f"Image: {image_path}")
    print(f"Prediction: {prediction}")
    print(f"Class probabilities:")

    with torch.no_grad():
        image_tensor = image_tensor.to(device).unsqueeze(0)
        probs = torch.softmax(model(image_tensor), dim=1)[0]
        for i, (label, prob) in enumerate(zip(label_encoder.classes_, probs)):
            print(f"  {label}: {prob.item():.2%}")

    return prediction


def main():
    parser = argparse.ArgumentParser(description="Animal Face Classifier Inference")
    parser.add_argument("image", help="Path to the input image")
    parser.add_argument(
        "--model",
        "-m",
        default=config.MODEL_PATH,
        help=f"Path to model file (default: {config.MODEL_PATH})",
    )
    parser.add_argument(
        "--data",
        "-d",
        default=config.DATA_PATH,
        help=f"Path to dataset (default: {config.DATA_PATH})",
    )
    args = parser.parse_args()

    infer(args.image, args.model, args.data)


if __name__ == "__main__":
    main()
