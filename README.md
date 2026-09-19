# 🧠 Deep Learning Journey

Welcome to my Deep Learning journey.
This repository documents my completed Deep Learning course: notes, practice problems, assignments, experiments, applied PyTorch projects, and exams.

---

## Overview

The course ran for six weeks and moved from a single perceptron to multi-layer artificial neural networks trained in PyTorch, then to techniques that make that training more stable and effective.

Everything is organized by week and module. A typical module contains lecture PDFs or slides, one or more Jupyter/Colab notebooks (course versions are prefixed `Phitron_`), and a `Practice Problem` folder with my written solutions.

---

## Learning Roadmap

```text
Week 01  Perceptrons
   ↓
Week 02  Multi-Layer Perceptrons
   ↓
Week 03  Backpropagation and Autograd
   ↓
Week 04  PyTorch Training Workflow
   ↓
Week 05  Applied ANN Projects
   ↓
Week 06  Scaling, Batch Normalization, Optimizers
```

---

## Repository Structure

```text
Deep-Learning/
├── Week 01 Perceptron Fundamentals and Training Concepts/
│   ├── Module 01  Introduction to Deep Learning and Perceptrons/
│   ├── Module 02 How Perceptrons Learn - Intuition, Code, and Edge Cases/
│   └── Module 03  Decision Boundaries and Perceptron Learning/
├── Week 02 From Perceptron to Multi-Layer Perceptron/
│   ├── Module 04 Learning in Perceptrons From Gradient Descent to Loss Functions/
│   ├── Module 05 From Activation To Loss Function/
│   ├── Module 06 Foundations of Multi-level Layer Perceptrons and Forward Propagation/
│   └── Module 07 DL Assignment 01/
├── Week 03 Backpropagation & Implementaion/
│   ├── Module 08 Backpropagation Fundamentals/
│   ├── Module 09 Pytorch Basic and Backpropagation Implementation/
│   ├── Module 10 From Computation Graph to Autograd Implementation/
│   └── Conceptual Session/
├── Week 04 Fundamentals of the PyTorch Training Workflow/
│   ├── Module 11 Pytorch Training Pipeline From Scratch/
│   ├── Module 12 PyTorch nn Module & torch.optim/
│   ├── Module 13 Training Dynamics in PyTorch - Batch, SGD & Mini-Batch/
│   ├── Module 14 DL Mid Term Exam/
│   └── Conceptual Session/
├── Week 05 Project Week - Applied Deep Learning Project/
│   ├── Module 15 Deep Learning Project Setup (Dataset and DataLoader inPyTorch)/
│   ├── Module 16 From ANN Implementation to GPU Training with PyTorch/
│   ├── Module 17 Loss Functions and Training Challenges in Deep Neural Networks/
│   └── Conceptual Session/
├── Week 06 Improving Neural Network Training and Performance/
│   ├── Module 18 Data Scaling, Standardization and Normalization/
│   ├── Module 19 Batch Normalization/
│   ├── Module 20 Optimizers/
│   ├── Module 21 DL Final Exam/
│   └── Conceptual Session/
├── Optional/
│   └── Week 01  Perceptron Fundamentals and Training Concepts/
└── README.md
```

Datasets kept in the repository: `fmnist_small.csv` (Week 05), `Social_Network_Ads.csv` (Week 06), and `.npy` image/label arrays (Week 04 Conceptual Session).

---

## Topics Covered

| Category | Topics |
| --- | --- |
| Perceptron fundamentals | Weights, bias, weighted sum, step function, weight updates, decision boundaries, AND/XOR, linear separability |
| Multi-layer networks | MLP notation, hidden layers, forward propagation, why non-linear activations matter |
| Activations | Sigmoid, Tanh, ReLU, Leaky ReLU, ELU |
| Loss functions | Binary Cross Entropy, Cross Entropy, regression loss, hinge loss (practice problem) |
| Optimization | Gradient descent (batch, stochastic, mini-batch), SGD, Adam, EWMA and optimizer comparison notes |
| Backpropagation | Manual gradient derivation, computation graphs, PyTorch Autograd |
| PyTorch workflow | Tensors, `nn.Module`, `nn.Sequential`, `torch.optim`, custom `Dataset`, `DataLoader`, train/eval loops, CPU and GPU training |
| Data preparation | Train/validation/test splits, standardization, normalization and MinMax scaling, label encoding |
| Training challenges | Vanishing gradients, effect of unscaled features, batch normalization |
| Evaluation | Accuracy, classification report, confusion matrix, learning curves |

---

## Technologies & Tools

| Category | Tools |
| --- | --- |
| Language | Python |
| Data and visualization | NumPy, Pandas, Matplotlib, Seaborn |
| Classical ML | scikit-learn |
| Deep learning | PyTorch (`torch`, `torch.nn`, `torch.optim`, `Dataset`/`DataLoader`, `torchvision`) |
| Supporting | `torchinfo`, `joblib`, `kagglehub` |
| TensorFlow/Keras | Used only in Week 01 (non-linear pattern comparison) and Week 02 (activation function demos) |
| Environment | Jupyter Notebook, Google Colab |
| Version control | Git / GitHub |

---

## Learning Modules

### Week 01: Perceptron Fundamentals and Training Concepts

Introduction to deep learning and the perceptron. Implemented a perceptron from scratch, worked through weight updates, and plotted AND and XOR decision boundaries to see where a single perceptron breaks down. A mini project classified the Iris dataset with scikit-learn's `Perceptron`. A separate notebook compared a linear model against a small neural network on a spiral (non-linear) dataset.

### Week 02: From Perceptron to Multi-Layer Perceptron

Gradient descent and loss functions, a detailed tour of activation functions, the advantages and limits of the perceptron, and MLP notation with forward propagation. Closes with Assignment 01.

### Week 03: Backpropagation and Implementation

Backpropagation worked out by hand on a small sigmoid network, including where gradients start to vanish. Then PyTorch tensors, backpropagation in PyTorch, computation graphs, and Autograd. The conceptual session ties backprop, Autograd, and a basic training pipeline together.

### Week 04: Fundamentals of the PyTorch Training Workflow

Built a full training pipeline from scratch on the Breast Cancer dataset, then refactored it using `nn.Module` and `torch.optim`. Compared batch, stochastic, and mini-batch gradient descent. The conceptual session trains a handwritten digit classifier from `.npy` arrays. Week ends with the midterm exam.

### Week 05: Project Week (Applied Deep Learning)

Custom `Dataset` and `DataLoader` classes, then a full Fashion-MNIST ANN trained on both CPU and GPU. Module 17 covers loss functions and training difficulties, including a vanishing gradient experiment. The conceptual session applies the same pipeline to Sign Language MNIST.

### Week 06: Improving Neural Network Training and Performance

Data scaling and its measurable effect on training, batch normalization applied to the Fashion-MNIST ANN, and optimizer theory covering EWMA and the different optimizer types available in PyTorch. Week ends with the final exam.

### Optional

Two extra Week 01 perceptron notebooks kept for additional practice.

---

## Practical Work & Experiments

| Work | Concepts | Description |
| --- | --- | --- |
| Perceptron from scratch, AND & XOR | Perceptron learning, decision boundaries | Manual weight updates; XOR shows the limits of a single layer |
| Iris mini project | Binary/multi-class classification | scikit-learn `Perceptron` on Iris features |
| Non-linear pattern comparison | Linear model vs neural network | Logistic regression and a small Keras MLP on a spiral dataset |
| Gradient descent notebooks | Gradient descent, regression loss | From-scratch gradient descent against scikit-learn baselines |
| Activation function notebooks | Sigmoid, Tanh, ReLU, Leaky ReLU, ELU | Curves, derivatives, and trade-offs of each activation |
| **NeuroNav (Assignment 01)** | Sigmoid, BCE, gradient descent | Forest vs Urban binary classification on `make_blobs` data, with a manually implemented forward pass, loss, training loop, and decision boundary plot |
| Backprop and Autograd notebooks | Chain rule, computation graphs | Hand-computed gradients verified against PyTorch Autograd |
| Breast Cancer training pipelines | End-to-end PyTorch workflow | Same problem rebuilt three times: from scratch, with `nn.Module`/`optim`, then with `Dataset`/`DataLoader` |
| Batch vs stochastic vs mini-batch | Training dynamics | Compared convergence behaviour of the three gradient descent variants |
| Handwritten digit classification (CS) | Training loop, ANN | Digit classifier trained from `.npy` image and label arrays |
| **Fashion-MNIST ANN, CPU and GPU** | `Dataset`, `DataLoader`, CUDA | Multi-layer ANN on `fmnist_small.csv`; CPU version then moved to GPU |
| **Sign Language MNIST (CS)** | ANN on flattened pixels | Fully connected network on 28x28 hand gesture images from CSV |
| **Vanishing gradient experiment** | Deep sigmoid networks | Deep network on `make_moons` showing gradients shrinking in early layers |
| **Social Network Ads scaling** | Standardization, MinMax scaling | Scaled vs unscaled training compared directly; scaling improved test accuracy from 0.675 to 0.800 (standard scaler) and 0.762 (MinMax) |
| Batch normalization on Fashion-MNIST | `BatchNorm1d` | Added batch norm layers to the GPU Fashion-MNIST model |

### Evaluations

| Exam | Task | Setup | Result |
| --- | --- | --- | --- |
| **Midterm (StyleAI)** | Fashion-MNIST clothing classification | `FashionClassifier` (`nn.Sequential`), CrossEntropyLoss, SGD, 15 epochs | **82.45%** test accuracy |
| **Final (TranslateAI)** | KMNIST ancient Japanese character recognition | `TranslateAIClassifier` with `BatchNorm1d`, CrossEntropyLoss, Adam, 15 epochs | **91.86%** test accuracy |

Both exams include learning curves and a confusion matrix with written analysis.

---

## Skills Developed

```text
✔ Perceptron theory and from-scratch implementation
✔ Choosing and reasoning about activation and loss functions
✔ Gradient descent variants: batch, stochastic, mini-batch
✔ Backpropagation by hand and with PyTorch Autograd
✔ Building models with nn.Module, nn.Sequential and torch.optim
✔ Custom Dataset and DataLoader pipelines
✔ Writing complete training and evaluation loops
✔ Moving training from CPU to GPU
✔ Feature scaling, standardization and normalization
✔ Applying batch normalization
✔ Recognizing and diagnosing vanishing gradients
✔ Evaluating models with accuracy, confusion matrices and learning curves
```

---

## How to Use

```bash
git clone https://github.com/ak1bhasan/Deep-Learning.git
cd Deep-Learning
```

1. Follow the weeks in order, Week 01 through Week 06, for the intended progression.
2. Open notebooks in Jupyter, VS Code, or Google Colab. Several were written in Colab and reference `/content/` paths, so adjust file paths when running locally.
3. Where both a `Phitron_*.ipynb` and a plain version exist, the `Phitron_*` file is the course version and the other is my own run.
4. PDFs and practice problem solutions in each module give the theory behind the code.

Typical dependencies:

```bash
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn
```

The exam notebooks download Fashion-MNIST and KMNIST through `torchvision`. Other notebooks read CSV or `.npy` files stored in their module folders.

---

## Learning Outcomes

By the end of the course I was able to:

- Explain and implement a perceptron, forward propagation, and backpropagation without a framework
- Build and train fully connected neural networks in PyTorch end to end
- Structure a reproducible pipeline: data, model, loss, optimizer, training loop, evaluation
- Move a training workflow from CPU to GPU
- Identify why a model trains poorly, whether from unscaled features or vanishing gradients, and apply scaling, batch normalization, or a better optimizer
- Evaluate results with accuracy, learning curves, and confusion matrices, and interpret what the errors mean

---

## Future Learning

Everything here uses fully connected networks. The natural next step is architectures that handle spatial and sequential structure, which were not part of this course.

---

## Acknowledgements

- **Phitron** for the Deep Learning course curriculum, notebooks, and practice problems
- ***Grokking Deep Learning*** by Andrew W. Trask, included as reference reading in Week 01

---

**Learn → Practice → Experiment → Build → Improve**
