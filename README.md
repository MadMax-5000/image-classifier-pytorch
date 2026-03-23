# 🐾 Animal Classifier

> Deep learning project for classifying **92 different animals** using CNNs with transfer learning and modern visualization techniques.

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📋 Overview

This project implements a complete image classification pipeline for classifying **92 different animal species** from multiple datasets. It supports both custom CNN architectures and pretrained models (ResNet, EfficientNet) with transfer learning. The project includes comprehensive visualizations including Grad-CAM attention maps and feature map analysis.

**Key Achievement**: 92% validation accuracy on 92 animal classes!

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| **92 Animal Classes** | Classifies 92 different animals (antelope to zebra) |
| **Multiple Architectures** | Custom CNN, ResNet18/34, EfficientNet-B0 |
| **Transfer Learning** | Pretrained ImageNet weights with fine-tuning |
| **Mixed Precision Training** | FP16 acceleration for faster GPU training |
| **Early Stopping** | Patience-based training with best checkpoint saving |
| **Grad-CAM** | Visualize what the CNN "looks at" for predictions |
| **Feature Maps** | Layer-by-layer activation visualization |
| **Data Augmentation** | Random flips, rotations, color jitter, affine transforms |
| **Confusion Matrix** | Comprehensive error analysis |
| **Top-K Predictions** | Shows multiple predictions with confidence percentages |
| **Gradio Interface** | Web-based inference demo |

---

## 🗂️ Project Structure

```
animal-classifier-pytorch/
├── config.py                 # All hyperparameters
├── main.py                   # Training entry point
├── inference.py              # CLI inference with top-K predictions
├── analyze_model.py          # Model analysis & visualization
├── confusion_matrix.py       # Confusion matrix generator
├── app.py                   # Gradio web interface
├── requirements.txt          # Dependencies
├── src/
│   ├── __init__.py
│   ├── model.py             # CNN architectures + factory
│   ├── train.py             # Training with mixed precision
│   ├── data.py              # Dataset loading & transforms
│   └── visualization.py      # Grad-CAM & feature maps
├── scripts/
│   ├── download_data.py      # Dataset download script
│   ├── consolidate_datasets.py # Merge multiple datasets
│   └── augment_data.py       # Balance class distribution
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

# Install PyTorch with CUDA (for GPU training)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

### 2. Consolidate Datasets (Optional)

```bash
# Run the consolidation script to merge multiple animal datasets
python scripts/consolidate_datasets.py --output data/consolidated

# Augment minority classes to balance the dataset
python scripts/augment_data.py --target 200 --data data/consolidated
```

### 3. Train Model

```bash
# EfficientNet-B0 (recommended - best accuracy)
python main.py --model efficientnet_b0 --epochs 25 --batch-size 32

# ResNet18 (faster training)
python main.py --model resnet18 --epochs 20

# Custom CNN (for experimentation)
python main.py --model custom
```

### 4. Run Inference

```bash
# CLI inference with top-5 predictions
python inference.py /path/to/image.jpg --top-k 5

# Web interface
python app.py
```

---

## 🎯 92 Animal Classes

The model classifies images into these categories:

**Mammals**: antelope, badger, bat, bear, bison, boar, camel, cat, cattle, cheetah, chimpanzee, coyote, deer, dog, dolphin, donkey, elephant, fox, giant_panda, goat, gorilla, hamster, hare, hedgehog, horse, hummmingbird, hyena, jaguar, jellyfish, kangaroo, koala, leopard, lion, lizard, lobster, monkey, moth, mouse, octopus, otter, owl, ox, panda, pig, pigeon, polar_bear, porcupine, possum, rabbit, raccoon, rat, reindeer, rhinoceros, sea_lion, sea_urchin, seal, shark, sheep, shrimps, skunk, snake, spider, squirrel, starfish, swan, tiger, turtle, whale, wolf, wombat, zebra

**Birds**: chicken, crow, duck, eagle, flamingo, goose, ostrich, owl, parakeet, penguin, pigeon, sea_gull, skimmer, swan, turkey

**Insects & Reptiles**: butterfly, caterpillar, cockroach, crab, dragonfly, fish, fly, grasshopper, ladybug, lizard, scorpion, sea_urchin, snake, spider

---

## 📐 Mathematical Foundation

### Cross-Entropy Loss

For multi-class classification with $C$ classes:

$$\mathcal{L}_{CE} = -\sum_{c=1}^{C} y_c \log(\hat{y}_c)$$

### Softmax Activation

Converts logits to probabilities:

$$\hat{y}_c = \frac{e^{z_c}}{\sum_{k=1}^{C} e^{z_k}}$$

### Grad-CAM

$$L^c_{Grad-CAM} = ReLU\left(\sum_k \alpha_k^c \cdot A^k\right)$$

---

## ⚙️ Configuration

All hyperparameters in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `IMG_SIZE` | 224 | Input image size |
| `BATCH_SIZE` | 16 | Training batch size |
| `EPOCHS` | 25 | Maximum training epochs |
| `LR` | 1e-3 | Learning rate |
| `MODEL_NAME` | efficientnet_b0 | Model architecture |
| `PRETRAINED` | True | Use ImageNet weights |
| `FREEZE_BACKBONE` | False | Fine-tune entire network |
| `EARLY_STOPPING_PATIENCE` | 8 | Early stop threshold |
| `TOP_K_PREDICTIONS` | 5 | Number of predictions to show |
| `MIXED_PRECISION` | True | Enable FP16 training |

---

## 🎨 Data Augmentation

| Transform | Description |
|-----------|-------------|
| Resize | 224×224 pixels |
| RandomHorizontalFlip | 50% probability |
| RandomRotation | ±20 degrees |
| ColorJitter | brightness, contrast, saturation = 0.3 |
| RandomAffine | translate, scale variations |
| Normalize | ImageNet mean/std |

---

## 📊 Inference Output

Example output from `inference.py`:

```
Image: sample.jpg

Top 5 Predictions:
----------------------------------------
1. deer          ████████████████████ 52.3%
2. antelope      ███████████████░░░░░░ 31.5%
3. horse         ██████░░░░░░░░░░░░░░ 8.2%
4. goat          ███░░░░░░░░░░░░░░░░░ 4.1%
5. sheep         ██░░░░░░░░░░░░░░░░░░ 1.9%
----------------------------------------

Prediction: deer (52.3%)
```

---

## 📊 Outputs

After training, the following files are generated:

| File | Description |
|------|-------------|
| `best_model.pth` | Best checkpoint (lowest val loss) |
| `my_model.pth` | Final model weights |
| `feature_maps.png` | Feature activation visualization |
| `gradcam_samples.png` | Grad-CAM attention maps |
| `training_curves.png` | Training & validation curves |

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
tqdm
```

---

## 🎯 Model Comparison

| Model | Parameters | Expected Accuracy | Training Time |
|-------|------------|-------------------|--------------|
| Custom CNN | ~3.3M | 70-80% | Fast |
| ResNet18 | ~11.2M | 85-92% | Medium |
| ResNet34 | ~21.8M | 88-93% | Medium-Slow |
| EfficientNet-B0 | ~4.1M | 90-95% | Slow |

---

## 📚 Datasets

This project uses combined data from:
- **AFHQ** (Animal Faces HQ) - 3 classes
- **Animal Image Dataset - 90 Different Animals** - 90 classes
- **Animals10** - 10 Italian animal classes (translated)

After consolidation and augmentation:
- **92 unique animal classes**
- **~45,000+ images**
- **Balanced class distribution** (200-5000 images per class)

---

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

---

<div align="center">

**Built with PyTorch** 🔥

*For educational purposes*

</div>
