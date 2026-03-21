import torch
from torch import nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from typing import Tuple


def train_one_epoch(
    model, loader: DataLoader, criterion, optimizer, device: str
) -> Tuple[float, float]:
    model.train()
    total_loss = 0
    total_acc = 0
    num_samples = 0

    for inputs, labels in loader:
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        total_loss += loss.item() * len(labels)

        preds = torch.argmax(outputs, dim=1)
        total_acc += (preds == labels).sum().item()
        num_samples += len(labels)

        loss.backward()
        optimizer.step()

    return total_loss / num_samples, 100 * total_acc / num_samples


def validate(model, loader: DataLoader, criterion, device: str) -> Tuple[float, float]:
    model.eval()
    total_loss = 0
    total_acc = 0
    num_samples = 0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            total_loss += loss.item() * len(labels)

            preds = torch.argmax(outputs, dim=1)
            total_acc += (preds == labels).sum().item()
            num_samples += len(labels)

    return total_loss / num_samples, 100 * total_acc / num_samples


def evaluate(model, loader: DataLoader, criterion, device: str) -> Tuple[float, float]:
    return validate(model, loader, criterion, device)


def train(
    model,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    scheduler=None,
    device: str = "cpu",
    early_stopping_patience: int = 10,
    save_best_path: str = "best_model.pth",
    save_best_only: bool = True,
) -> dict:
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=lr)

    if scheduler is None:
        scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=3)

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
    }

    best_val_loss = float("inf")
    patience_counter = 0
    best_epoch = 0

    for epoch in range(epochs):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        scheduler.step(val_loss)

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            patience_counter = 0
            best_epoch = epoch + 1
            torch.save(model.state_dict(), save_best_path)
        else:
            patience_counter += 1

        status = ""
        if improved:
            status = " [BEST]"
        if patience_counter >= early_stopping_patience:
            status += " [EARLY STOPPING]"

        print(
            f"Epoch {epoch + 1}/{epochs} - "
            f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% - "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%{status}"
        )

        if patience_counter >= early_stopping_patience:
            print(f"Early stopping triggered at epoch {epoch + 1}")
            break

    history["best_epoch"] = best_epoch
    if save_best_only:
        print(f"Best model saved to {save_best_path} (epoch {best_epoch})")

    return history


def predict(model, image, device: str = "cpu"):
    model.eval()
    with torch.no_grad():
        image = image.to(device)
        output = model(image.unsqueeze(0))
        return torch.argmax(output, dim=1).item()
