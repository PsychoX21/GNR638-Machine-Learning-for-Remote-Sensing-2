# GNR638: Machine Learning for Remote Sensing II

This repository contains coursework, assignments, and projects for **GNR638: Machine Learning for Remote Sensing II** (Spring 2026).

## Repository Structure

### Assignment 1

**Objective**: Build a deep learning framework from scratch (C++ backend + Python frontend) without using PyTorch/TensorFlow.

- **Key Features**:
  - Custom C++ backend with CUDA acceleration.
  - Reverse-mode Automatic Differentiation (Autograd).
  - Implementation of ResNet-20 for MNIST and CIFAR-100 ($>99\%$ and $\sim70\%$ accuracy).
  - Optimized build system with `pybind11` integration.

### Assignment 2

**Objective**: Study pre-trained CNN representation transfer, fine-tuning strategies, and robustness under data constraints and input corruption.

- **Key Features**:
  - Comparative analysis of 5 CNN architectures (ResNet, Inception, DenseNet, EfficientNet, ConvNeXt).
  - Evaluation of fine-tuning strategies and automated sensitivity-based layer selection.
  - Robustness testing against Gaussian noise, motion blur, and brightness shifts.
  - Data efficiency analysis under few-shot (5% and 20%) settings.
  - Layer-wise feature probing to analyze representation quality across network depth.

---

**Collaborators**:

- Saksham Khandelwal (24B0965)
- Pawan Kumar Meena (24B0904)
- Rohan Jadhav (24B1012)

**Institute**: IIT Bombay
