import argparse
import torch
from torch.utils.data import DataLoader

import config
from src import (
    CustomImageDataset,
    create_model,
    evaluate,
    get_target_layer,
    get_transforms,
    load_data,
    plot_gradcam,
    split_data,
    visualize_feature_maps,
)


def main():
    parser = argparse.ArgumentParser(description="Analyze Trained Model")
    parser.add_argument(
        "--model", type=str, default=config.BEST_MODEL_PATH, help="Path to model file"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=config.MODEL_NAME,
        help="Model architecture name",
    )
    parser.add_argument(
        "--data", type=str, default=config.DATA_PATH, help="Path to dataset"
    )
    parser.add_argument(
        "--output", type=str, default="analysis", help="Output directory prefix"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    df, label_encoder = load_data(args.data)
    _, _, test_df = split_data(df, config.TRAIN_SPLIT, config.VAL_SPLIT)

    test_transform = get_transforms(config.IMG_SIZE, augment=False)
    test_dataset = CustomImageDataset(test_df, test_transform, label_encoder)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False)

    num_classes = len(label_encoder.classes_)
    print(f"Classes: {list(label_encoder.classes_)}")

    model = create_model(args.model_name, num_classes, pretrained=False).to(device)

    try:
        model.load_state_dict(torch.load(args.model, weights_only=True))
        print(f"Loaded model from {args.model}")
    except Exception as e:
        print(f"Could not load model: {e}")
        return

    test_loss, test_acc = evaluate(
        model, test_loader, torch.nn.CrossEntropyLoss(), device
    )
    print(f"\nTest Results:")
    print(f"  Accuracy: {test_acc:.2f}%")
    print(f"  Loss: {test_loss:.4f}")

    sample_images = [test_dataset[i][0] for i in range(min(3, len(test_dataset)))]
    sample_targets = [
        test_dataset[i][1].item() for i in range(min(3, len(test_dataset)))
    ]

    print(f"\nGenerating visualizations...")

    try:
        visualize_feature_maps(
            model,
            sample_images[0].to(device),
            save_path=f"{args.output}_feature_maps.png",
            device=device,
        )
    except Exception as e:
        print(f"Feature map generation failed: {e}")

    try:
        target_layer = get_target_layer(model, args.model_name)
        if target_layer is not None:
            plot_gradcam(
                model,
                sample_images[:3],
                sample_targets[:3],
                list(label_encoder.classes_),
                target_layer,
                save_path=f"{args.output}_gradcam.png",
                device=device,
            )
    except Exception as e:
        print(f"Grad-CAM generation failed: {e}")

    print(f"\nAnalysis complete!")
    print(f"  Feature maps: {args.output}_feature_maps.png")
    print(f"  Grad-CAM: {args.output}_gradcam.png")


if __name__ == "__main__":
    main()
