# ML @ Berkeley — Coursework

Notebooks completed for an introductory deep-learning course (Fall 2025), adapted from Stanford's CS231N teaching materials. Each notebook is self-contained and runnable in Colab.

---

## Contents

| Notebook | Topic |
|---|---|
| [`hw1-intro-to-pytorch.ipynb`](hw1-intro-to-pytorch.ipynb) | PyTorch fundamentals: tensors, autograd, datasets, and three ways to build a network |

More assignments will be added here as the course progresses.

---

## HW1 — Introduction to PyTorch

PyTorch behaves like NumPy but adds automatic differentiation, letting you write the forward pass of a network and get gradients for free instead of deriving and coding backprop by hand. The assignment builds up from raw tensors to a trained model in three progressively more abstract ways.

**Covered:**
1. Tensors — NumPy/PyTorch parity (creation, indexing, reshaping, linear algebra)
2. Autograd — automatic differentiation
3. Datasets & `DataLoader` — batching and on-the-fly augmentation
4. Barebones PyTorch — manual forward pass and manual weight updates
5. `nn.Module` — the standard way to define a model
6. `nn.Sequential` — the concise way, for simple architectures

**Dataset pipeline & augmentation**

A `DataLoader` applying a random-rotation transform on every read, so the same source image comes out differently each epoch:

![Dataloader augmentation example](img/hw1_dataloader_augmentation.png)

**Fitting a nonlinear function**

An early barebones-PyTorch exercise: fit a small network to a noisy `sin(x)` dataset using nothing but tensors and manual gradient updates.

![Sin curve dataset](img/hw1_sin_dataset.png)

---

## Setup

```bash
pip install torch torchvision numpy matplotlib
```

Open any notebook in Jupyter or Colab (badges are included at the top of each notebook) and run top to bottom.
