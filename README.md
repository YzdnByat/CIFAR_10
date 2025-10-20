# CIFAR-10 Image Classification (PyTorch)

This repository implements image classification on the
[CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset using **PyTorch**.
It includes a compact CNN baseline and a modular training/evaluation pipeline with
augmentation, learning curves, and a confusion matrix.

---

## ✨ Features

- **Dataset & Augmentation**
  - `torchvision.datasets.CIFAR10`
  - Normalization `(0.4914, 0.4822, 0.4465)` / `(0.2023, 0.1994, 0.2010)`
  - `RandomCrop(32, padding=4)` and `RandomHorizontalFlip()` for training
- **Model**
  - 3-block CNN (`Conv → BatchNorm → ReLU → MaxPool`) with channels `32 → 64 → 128`
  - Linear head: `128 × 4 × 4 → 10`
- **Training**
  - `Adam` optimizer, `CrossEntropyLoss`
  - Configurable `--epochs`, `--batch-size`, `--lr`
- **Evaluation & Visualization**
  - Accuracy per epoch, final test accuracy
  - Classification report & **confusion matrix**
  - Loss/accuracy curves saved to `results/`
