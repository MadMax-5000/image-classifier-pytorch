import argparse
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from torchsummary import summary

import config
from src import (
    CustomImageDataset,
    create_model,
    evaluate,
    freeze_backbone,
    get_target_layer,
    get_transforms,
    load_data,
    plot_gradcam,
    predict,
    split_data,
    train,
    visualize_feature_maps,
)


def visualize_samples(df):
    n_rows, n_cols = 3, 3
    fig, axarr = plt.subplots(n_rows, n_cols)
    for row in range(n_rows):
        for col in range(n_cols):
            image_path = df.sample(n=1)["image_path"].iloc[0]
            image = Image.open(image_path).convert("RGB")
            axarr[row, col].imshow(image)
            axarr[row, col].axis("off")
    plt.show()


def plot_training_curves(history):
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))

    axs[0].plot(history["train_loss"], label="Train Loss")
    axs[0].plot(history["val_loss"], label="Val Loss")
    axs[0].set_title("Training and Validation Loss over Epochs")
    axs[0].set_xlabel("Epochs")
    axs[0].set_ylabel("Loss")
    axs[0].legend()

    axs[1].plot(history["train_acc"], label="Train Accuracy")
    axs[1].plot(history["val_acc"], label="Val Accuracy")
    axs[1].set_title("Training and Validation Accuracy over Epochs")
    axs[1].set_xlabel("Epochs")
    axs[1].set_ylabel("Accuracy")
    axs[1].legend()
    plt.show()


def generate_visualizations(model, test_dataset, label_encoder, model_name, device):
    print("\nGenerating visualizations...")

    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
    images = []
    targets = []

    for img, label in test_loader:
        if len(images) >= 3:
            break
        images.append(img[0])
        targets.append(label[0].item())

    sample_images = [test_dataset[i][0] for i in range(min(3, len(test_dataset)))]
    sample_targets = [
        test_dataset[i][1].item() for i in range(min(3, len(test_dataset)))
    ]

    try:
        visualize_feature_maps(
            model,
            sample_images[0].to(device),
            save_path="feature_maps.png",
            device=device,
        )
    except Exception as e:
        print(f"Could not generate feature maps: {e}")

    try:
        target_layer = get_target_layer(model, model_name)
        if target_layer is not None:
            plot_gradcam(
                model,
                sample_images[:3],
                sample_targets[:3],
                list(label_encoder.classes_),
                target_layer,
                save_path="gradcam_samples.png",
                device=device,
            )
    except Exception as e:
        print(f"Could not generate Grad-CAM: {e}")


def main():
    parser = argparse.ArgumentParser(description="Animal Face Classifier")
    parser.add_argument(
        "--model",
        type=str,
        default=config.MODEL_NAME,
        choices=["custom", "resnet18", "resnet34", "efficientnet_b0"],
        help="Model architecture to use",
    )
    parser.add_argument(
        "--epochs", type=int, default=config.EPOCHS, help="Number of training epochs"
    )
    parser.add_argument(
        "--no-train", action="store_true", help="Skip training, just run visualizations"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    print(f"Model: {args.model}")

    df, label_encoder = load_data(config.DATA_PATH)
    train_df, val_df, test_df = split_data(df, config.TRAIN_SPLIT, config.VAL_SPLIT)

    train_transform = get_transforms(config.IMG_SIZE, augment=True)
    test_transform = get_transforms(config.IMG_SIZE, augment=False)

    train_dataset = CustomImageDataset(train_df, train_transform, label_encoder)
    val_dataset = CustomImageDataset(val_df, test_transform, label_encoder)
    test_dataset = CustomImageDataset(test_df, test_transform, label_encoder)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=True)

    if not args.no_train:
        visualize_samples(df)

    num_classes = len(df["labels"].unique())
    print(f"Number of classes: {num_classes}")
    print(f"Classes: {list(label_encoder.classes_)}")

    model = create_model(args.model, num_classes, config.PRETRAINED, config.DROPOUT).to(
        device
    )

    try:
        summary(model, input_size=(3, config.IMG_SIZE, config.IMG_SIZE))
    except Exception as e:
        print(f"Could not print model summary: {e}")
        print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")

    if config.FREEZE_BACKBONE and args.model != "custom":
        print("Freezing backbone (transfer learning mode)")
        freeze_backbone(model, args.model)

    if not args.no_train:
        scheduler = ReduceLROnPlateau(
            torch.optim.Adam(model.parameters(), lr=config.LR),
            mode="min",
            factor=config.SCHEDULER_FACTOR,
            patience=config.PATIENCE,
        )

        history = train(
            model,
            train_loader,
            val_loader,
            args.epochs,
            config.LR,
            scheduler,
            device,
            early_stopping_patience=config.EARLY_STOPPING_PATIENCE,
            save_best_path=config.BEST_MODEL_PATH,
            save_best_only=config.SAVE_BEST_ONLY,
        )

        model.load_state_dict(torch.load(config.BEST_MODEL_PATH, weights_only=True))
        print(f"Loaded best model from {config.BEST_MODEL_PATH}")

        torch.save(model.state_dict(), config.MODEL_PATH)
        print(f"Final model saved to {config.MODEL_PATH}")

        test_loss, test_acc = evaluate(
            model, test_loader, torch.nn.CrossEntropyLoss(), device
        )
        print(f"Test Accuracy: {test_acc:.2f}%, Test Loss: {test_loss:.4f}")

        plot_training_curves(history)

    generate_visualizations(model, test_dataset, label_encoder, args.model, device)


def predict_image(
    image_path: str,
    model_path: str = config.BEST_MODEL_PATH,
    model_name: str = config.MODEL_NAME,
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    df, label_encoder = load_data(config.DATA_PATH)
    test_transform = get_transforms(config.IMG_SIZE, augment=False)

    num_classes = len(df["labels"].unique())
    model = create_model(model_name, num_classes, pretrained=False).to(device)
    model.load_state_dict(torch.load(model_path, weights_only=True))
    model.eval()

    image = Image.open(image_path).convert("RGB")
    image_tensor = test_transform(image)

    pred = predict(model, image_tensor, device)
    return label_encoder.inverse_transform([pred])[0]


if __name__ == "__main__":
    main()
