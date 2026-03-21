# 🐾 Animal Face Classification

> Deep learning project for classifying animal faces (cat, dog, wild) using CNNs with transfer learning and modern visualization techniques.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Overview

This project implements a complete image classification pipeline for the **AFHQ (Animal Faces HQ)** dataset. It supports both custom CNN architectures and pretrained models (ResNet, EfficientNet) with transfer learning. The project includes comprehensive visualizations including Grad-CAM attention maps and feature map analysis.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **Multiple Architectures** | Custom CNN, ResNet18/34, EfficientNet-B0 |
| **Transfer Learning** | Pretrained ImageNet weights with fine-tuning |
| **Early Stopping** | Patience-based training with best checkpoint saving |
| **Grad-CAM** | Visualize what the CNN "looks at" for predictions |
| **Feature Maps** | Layer-by-layer activation visualization |
| **Data Augmentation** | Random flips, rotations, color jitter |
| **Confusion Matrix** | Comprehensive error analysis |
| **Gradio Interface** | Web-based inference demo |

---

## 🗂️ Project Structure

```
animal-classifier-pytorch/
├── config.py                 # All hyperparameters
├── main.py                   # Training entry point
├── inference.py              # CLI inference script
├── analyze_model.py          # Model analysis & visualization
├── confusion_matrix.py       # Confusion matrix generator
├── app.py                    # Gradio web interface
├── requirements.txt          # Dependencies
├── src/
│   ├── __init__.py
│   ├── model.py              # CNN architectures + factory
│   ├── train.py              # Training with early stopping
│   ├── data.py               # Dataset loading & transforms
│   └── visualization.py      # Grad-CAM & feature maps
├── scripts/
│   └── download_data.py      # Dataset download script
├── tests/
│   ├── test_model.py         # Model unit tests
│   ├── test_training.py       # Training tests
│   └── test_data.py          # Data pipeline tests
└── data/                     # Dataset location
```

---

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
.\venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Download Dataset

```bash
python scripts/download_data.py --output data --verify
```

### 3. Train Model

```bash
# Default: ResNet18 with pretrained weights
python main.py

# Custom CNN (faster, lower accuracy)
python main.py --model custom

# EfficientNet-B0
python main.py --model efficientnet_b0

# Fewer epochs
python main.py --epochs 10
```

### 4. Run Inference

```bash
# CLI inference
python inference.py /path/to/image.jpg

# Web interface
python app.py
```

---

## 📐 Mathematical Foundation

### Convolutional Layer

A convolution operation applies a filter $K$ to input $X$:

<p align="center"><font size="4">

$$Y_{i,j} = \sum_{m}\sum_{n} X_{i+m, j+n} \cdot K_{m,n}$$

</font></p>

Where:
- $Y$ = output feature map
- $K$ = learnable kernel/filter
- $i, j$ = spatial positions

### Batch Normalization

Normalizes layer inputs to have zero mean and unit variance:

<p align="center"><font size="4">

$$\hat{x} = \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}}$$

$$y = \gamma\hat{x} + \beta$$

</font></p>

Where $\gamma, \beta$ are learnable scale and shift parameters.

### Cross-Entropy Loss

For multi-class classification with $C$ classes:

<p align="center"><font size="4">

$$\mathcal{L}_{CE} = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)$$

</font></p>

Where:
- $y_c$ = ground truth label (one-hot)
- $\hat{y}_c$ = predicted probability for class $c$

### Softmax Activation

Converts logits to probabilities:

<p align="center"><font size="4">

$$\hat{y}_c = \frac{e^{z_c}}{\sum_{k=1}^{C} e^{z_k}}$$

</font></p>

### Backpropagation

Weights updated via gradient descent:

<p align="center"><font size="4">

$$w_{new} = w_{old} - \eta \cdot \frac{\partial \mathcal{L}}{\partial w}$$

</font></p>

Where $\eta$ = learning rate.

### Early Stopping

Training stops when validation loss doesn't improve for $p$ epochs:

<p align="center"><font size="4">

$$\text{stop if } \exists p \in [1, \text{patience}]: \mathcal{L}_{val}(t-p) < \mathcal{L}_{val}(t)$$

</font></p>

---

## ⚙️ Configuration

All hyperparameters in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMG_SIZE` | 224 | Input image size |
| `BATCH_SIZE` | 16 | Training batch size |
| `EPOCHS` | 15 | Maximum training epochs |
| `LR` | 1e-3 | Learning rate |
| `MODEL_NAME` | resnet18 | Model architecture |
| `PRETRAINED` | True | Use ImageNet weights |
| `FREEZE_BACKBONE` | True | Freeze pretrained layers |
| `EARLY_STOPPING_PATIENCE` | 10 | Early stop threshold |
| `DROPOUT` | 0.5 | Regularization rate |

---

## 🏗️ Model Architectures

### Custom CNN
```
Conv2d(3, 32) → BatchNorm → ReLU → MaxPool
Conv2d(32, 64) → BatchNorm → ReLU → MaxPool
Conv2d(64, 128) → BatchNorm → ReLU → MaxPool
Flatten → Dropout → Linear(128*16*16, 128) → Linear(128, num_classes)
```

### ResNet18 (Transfer Learning)
```
Input → ResNet18 Backbone (pretrained) → Modified FC Layer → Output
```

---

## 🎨 Data Augmentation

| Transform | Description |
|-----------|-------------|
| Resize | 224×224 pixels |
| RandomHorizontalFlip | 50% probability |
| RandomRotation | ±15 degrees |
| ColorJitter | brightness, contrast, saturation = 0.2 |
| Normalize | ImageNet mean/std |

---

## 📊 Outputs

After training, the following files are generated:

| File | Description |
|------|-------------|
| `best_model.pth` | Best checkpoint (lowest val loss) |
| `my_model.pth` | Final model weights |
| `feature_maps.png` | Feature activation visualization |
| `gradcam_samples.png` | Grad-CAM attention maps |
| `plots.png` | Training curves |

---

## 🔬 Visualization

### Grad-CAM (Gradient-weighted Class Activation Mapping)

Grad-CAM highlights regions in the image that are important for the CNN's prediction:

<p align="center"><font size="4">

$$L^c_{Grad-CAM} = ReLU\left(\sum_k \alpha_k^c \cdot A^k\right)$$

</font></p>

Where:
- $A^k$ = feature map activations
- $\alpha_k^c$ = gradient weights for class $c$

### Feature Maps

Visualize how different convolutional layers respond to input, showing edge detection, texture patterns, and complex feature combinations.

---

## 🧪 Testing

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_model.py

# Run with coverage
pytest tests/ --cov=src
```

---

## 📦 Dependencies

```
torch>=2.0.0
torchvision>=0.15.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
pandas>=2.0.0
numpy>=1.24.0
Pillow>=10.0.0
torchsummary>=1.5.1
pytest>=7.4.0
kagglehub>=0.2.0
gradio>=4.0.0
seaborn>=0.12.0
```

---

## 🎯 Model Comparison

| Model | Parameters | Expected Accuracy | Training Time |
|-------|------------|-------------------|--------------|
| Custom CNN | ~3.3M | 70-80% | Fast |
| ResNet18 | ~11.2M | 85-95% | Medium |
| EfficientNet-B0 | ~5.3M | 90-97% | Slow |

---

## 📚 Dataset

**AFHQ (Animal Faces HQ)** by Andrew M. on Kaggle

- 15,000+ images across 3 categories
- Cats, Dogs, Wild animals
- High-resolution (512×512)
- URL: https://www.kaggle.com/datasets/andrewmvd/animal-faces

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

---

<div align="center">

**Built with PyTorch** 🔥

*For educational purposes*

</div>
