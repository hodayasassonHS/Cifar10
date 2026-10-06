# CIFAR-10 Image Classification

Image classification on the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) dataset (60,000 32×32 colour images, 10 classes) using PyTorch.

## What's inside

All the work is in [`cifar10.ipynb`](cifar10.ipynb):

1. **Data loading** and exploratory analysis (per-class colour histograms).
2. **Preprocessing** – data augmentation (rotation, horizontal flip, random resized crop) and an 80/20 train/validation split.
3. **Custom CNN** – 4 convolutional layers with batch normalization and dropout (~18.6M trainable parameters), trained with Adam and a `ReduceLROnPlateau` scheduler.
4. **Transfer learning** – a pre-trained ResNet50 with a custom classifier head, for comparison.
5. **Test-set evaluation** – accuracy, classification report and confusion matrix.
6. **Inference** on a single external image.

## Results

| Model | Epochs | Validation accuracy | Test accuracy |
| --- | --- | --- | --- |
| Custom CNN | 140 | 88.2% | 89.5% |
| ResNet50 + custom head | stopped at 79 | 84.5% | – |

The custom CNN is also evaluated on the held-out 10,000-image test set, with a per-class precision/recall report and a confusion matrix (see the notebook). The ResNet50 comparison was stopped early and not evaluated on the test set.

## Tech stack

Python, PyTorch, torchvision, scikit-learn, NumPy, OpenCV, Matplotlib

## Running it

```bash
pip install -r requirements.txt
jupyter notebook cifar10.ipynb
```

The dataset is downloaded automatically by torchvision on first run. The inference example at the end of the notebook uses a local image path, so point it at your own file.
