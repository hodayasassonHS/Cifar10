# CIFAR-10 Image Classification

Image classification on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset (60,000 32×32 colour images, 10 classes) using PyTorch.

## What's inside

All the work is in [`cifar10.ipynb`](cifar10.ipynb):

1. **Data loading** and exploratory analysis (per-class colour histograms).
2. **Preprocessing** – data augmentation (rotation, horizontal flip, random resized crop) and an 80/20 train/validation split.
3. **Custom CNN** – 4 convolutional layers with batch normalization and dropout (~18.6M trainable parameters), trained with Adam and a `ReduceLROnPlateau` scheduler.
4. **Transfer learning** – a pre-trained ResNet50 with a custom classifier head, for comparison.
5. **Inference** on a single external image.

## Results

| Model | Epochs | Validation accuracy |
| --- | --- | --- |
| Custom CNN | 140 | 88.2% |
| ResNet50 + custom head | stopped at 79 | 84.5% |

Accuracy is measured on the validation split, not on the held-out test set.

## Tech stack

Python, PyTorch, torchvision, NumPy, OpenCV, Matplotlib

## Running it

```bash
pip install torch torchvision numpy opencv-python matplotlib scipy pillow
jupyter notebook cifar10.ipynb
```

The dataset is downloaded automatically by torchvision on first run. The inference example at the end of the notebook uses a local image path, so point it at your own file.
