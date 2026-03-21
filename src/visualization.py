import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import List, Optional
from PIL import Image


class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []

        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_cam(
        self, input_tensor: torch.Tensor, target_class: Optional[int] = None
    ) -> np.ndarray:
        self.model.eval()
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)

        self.model.zero_grad()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        one_hot = torch.zeros_like(output)
        one_hot[0, target_class] = 1
        output.backward(gradient=one_hot, retain_graph=True)

        if self.gradients is None or self.activations is None:
            return np.zeros((1, 1))

        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])

        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]

        heatmap = torch.mean(self.activations, dim=1).squeeze()
        heatmap = torch.relu(heatmap)
        max_val = heatmap.max()
        if max_val > 0:
            heatmap = heatmap / max_val

        return heatmap.cpu().numpy()

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()


def overlay_cam_on_image(
    image: Image.Image,
    cam: np.ndarray,
    alpha: float = 0.4,
) -> Image.Image:
    cam = cam - cam.min()
    if cam.max() > 0:
        cam = cam / cam.max()

    cam = np.uint8(255 * cam)
    cam_pil = Image.fromarray(cam).resize(image.size, Image.Resampling.BILINEAR)
    cam_array = np.array(cam_pil)

    jet_cmap = cm.ScalarMappable(cmap="jet")
    jet = jet_cmap.to_rgba(cam_array)[:, :, :3]
    jet = np.uint8(255 * jet)

    img_array = np.array(image).astype(np.float32) / 255
    jet_float = jet.astype(np.float32) / 255

    overlay = (1 - alpha) * img_array + alpha * jet_float
    overlay = np.clip(overlay, 0, 1)
    overlay = (overlay * 255).astype(np.uint8)

    return Image.fromarray(overlay)


def plot_gradcam(
    model: nn.Module,
    images: List[torch.Tensor],
    target_classes: List[int],
    class_names: List[str],
    target_layer: nn.Module,
    save_path: str = "gradcam_results.png",
    device: str = "cpu",
):
    n = len(images)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))
    if n == 1:
        axes = axes.reshape(2, -1)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    for i, (img_tensor, target_class) in enumerate(zip(images, target_classes)):
        img_tensor = img_tensor.to(device)

        gradcam = GradCAM(model, target_layer)
        cam = gradcam.generate_cam(img_tensor, target_class)
        gradcam.remove_hooks()

        original_img = img_tensor.cpu().detach().clone()
        original_img = original_img * std + mean
        original_img = torch.clamp(original_img, 0, 1)
        original_img = original_img.permute(1, 2, 0).numpy()

        overlaid = overlay_cam_on_image(
            Image.fromarray((original_img * 255).astype(np.uint8)), cam
        )

        axes[0, i].imshow(original_img)
        axes[0, i].set_title(f"Original\n{class_names[target_class]}")
        axes[0, i].axis("off")

        axes[1, i].imshow(overlaid)
        axes[1, i].set_title(f"Grad-CAM\n{class_names[target_class]}")
        axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Grad-CAM visualization saved to {save_path}")
    return fig


def visualize_feature_maps(
    model: nn.Module,
    image_tensor: torch.Tensor,
    save_path: str = "feature_maps.png",
    num_maps: int = 16,
    device: str = "cpu",
):
    model.eval()
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    hooks = []
    features = {}

    def get_features(name):
        def hook_fn(module, input, output):
            features[name] = output.detach()

        return hook_fn

    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append((name, module))

    if not conv_layers:
        print("No convolutional layers found")
        return None

    for name, module in conv_layers:
        hooks.append(module.register_forward_hook(get_features(name)))

    with torch.no_grad():
        _ = model(image_tensor)

    for hook in hooks:
        hook.remove()

    if not features:
        print("Could not extract feature maps")
        return None

    layer_names = list(features.keys())[:3]
    n_layers = len(layer_names)

    fig, axes = plt.subplots(
        n_layers, num_maps, figsize=(num_maps * 1.5, n_layers * 1.5)
    )
    if n_layers == 1:
        axes = axes.reshape(1, -1)

    for idx, layer_name in enumerate(layer_names):
        feat = features[layer_name][0].cpu().numpy()

        n_channels = min(num_maps, feat.shape[0])
        for ch in range(n_channels):
            axes[idx, ch].imshow(feat[ch], cmap="viridis")
            axes[idx, ch].axis("off")
            if ch == 0:
                axes[idx, ch].set_title(layer_name, fontsize=8)

        for ch in range(n_channels, num_maps):
            axes[idx, ch].axis("off")

    plt.suptitle("Feature Map Activations", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Feature maps saved to {save_path}")
    return fig


def visualize_single_feature_maps(
    model: nn.Module,
    image_tensor: torch.Tensor,
    layer_indices: List[int] = None,
    save_path: str = "feature_maps_detailed.png",
    device: str = "cpu",
):
    model.eval()
    if image_tensor.dim() == 3:
        image_tensor = image_tensor.unsqueeze(0)
    image_tensor = image_tensor.to(device)

    hooks = []
    features = {}

    def get_features(name):
        def hook_fn(module, input, output):
            features[name] = output.detach()

        return hook_fn

    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append((name, module))

    if not conv_layers:
        print("No convolutional layers found")
        return None

    for name, module in conv_layers:
        hooks.append(module.register_forward_hook(get_features(name)))

    with torch.no_grad():
        _ = model(image_tensor)

    for hook in hooks:
        hook.remove()

    feature_list = list(features.items())
    if layer_indices:
        valid_indices = [i for i in layer_indices if i < len(feature_list)]
        feature_list = [feature_list[i] for i in valid_indices]

    if not feature_list:
        print("No feature maps extracted")
        return None

    fig, axes = plt.subplots(len(feature_list), 8, figsize=(16, len(feature_list) * 2))
    if len(feature_list) == 1:
        axes = axes.reshape(1, -1)

    for row, (name, feat) in enumerate(feature_list):
        feat = feat[0].cpu().numpy()
        n_channels = min(8, feat.shape[0])

        for ch in range(n_channels):
            axes[row, ch].imshow(feat[ch], cmap="viridis")
            axes[row, ch].axis("off")
            if ch == 0:
                axes[row, ch].set_ylabel(name[:20], fontsize=8)

        for ch in range(n_channels, 8):
            axes[row, ch].axis("off")

    plt.suptitle("Layer-wise Feature Map Activations", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"Feature maps saved to {save_path}")
    return fig
