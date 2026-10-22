# 🧠 Deep Learning Journey

This repository is my record of the Deep Learning course at [Phitron](https://phitron.io). It holds the course notes, the instructor notebooks, my own re-runs, practice-problem solutions, one assignment, the conceptual sessions, and the midterm and final exams.

The work starts with a single perceptron and ends with fully connected networks trained in PyTorch: data pipelines, CPU and GPU training, scaling, batch normalization, and optimizer notes. All 21 modules are in the repository. A few practice sheets have questions only, and Module 20 is notes without code. Image tasks in this course flatten each picture into a vector and pass it through linear layers. There are no convolutional or recurrent models here.

Numbers below are the values saved in the notebooks. I did not retrain anything while writing this README.

---

## Overview

The course is six weeks long. Each week is split into numbered modules, and some weeks also have a conceptual session.

Inside a module you will usually find some mix of:

| File type | What it is |
|---|---|
| `*.pdf`, `*.pptx`, `*.docx` | Course notes, often exported from OneNote. Many of those PDFs are image pages, so little text can be extracted from them. The `.docx` files are mostly link lists. |
| `Phitron_*.ipynb` | The instructor notebook, kept under its original name |
| Unprefixed `*.ipynb` or `*.md` | My own run of the same material, or my written solution |

The `Phitron_` prefix is consistent where a pair exists: drop the prefix and you are looking at my copy. It is not true that every notebook has a twin. The spiral comparison, the tensor tour, both exams, the sign-language session, the Week 06 scaling sessions, and the batch-normalization notebook exist as only one file. Module 03 has no `.ipynb` at all.

---

## Learning Roadmap

```text
Week 01  Perceptron fundamentals
   ├─ Weights, bias, the step function, and a from-scratch perceptron
   ├─ AND and OR are learnable; XOR is not
   ├─ Decision boundaries, and how weight versus bias moves the line
   └─ Iris (setosa vs. the rest) and a spiral that a linear model cannot cut
          │
Week 02  From the perceptron to a multi-layer network
   ├─ Gradient descent on a regression line, and the perceptron loss
   ├─ Sigmoid, tanh, ReLU, and the other activations in the course notebook
   ├─ Why stacking linear layers without an activation collapses
   └─ Assignment 01 (NeuroNav): sigmoid, binary cross-entropy, a manual training loop
          │
Week 03  Backpropagation
   ├─ One full hand-worked backward pass, including shrinking gradients
   ├─ PyTorch tensors, then a small network updated by hand
   ├─ Computation graphs and Autograd, including the errors you get without requires_grad
   └─ Conceptual session: NumPy backprop, Autograd, and a tiny nn.Module loop on XOR
          │
Week 04  The PyTorch training workflow
   ├─ Breast Cancer logistic regression written without nn.Module, then with it
   ├─ Batch gradient descent versus one-example SGD on a 50-point toy set
   ├─ Conceptual session: a digit network on the included .npy arrays
   └─ Midterm (StyleAI): Fashion-MNIST with a fully connected network
          │
Week 05  Applied projects
   ├─ Custom Dataset and DataLoader, first on Breast Cancer again
   ├─ A Fashion-MNIST-style ANN on CPU, then the same idea on GPU
   ├─ A deep sigmoid net on make_moons whose loss barely moves
   └─ Conceptual session: Sign Language MNIST, flattened, 24 letters
          │
Week 06  Making training behave
   ├─ Standardization and min-max scaling on Social Network Ads
   ├─ Batch normalization added to the Fashion-MNIST ANN
   ├─ Optimizer notes: momentum, AdaGrad, RMSProp, Adam
   └─ Final exam (TranslateAI): KMNIST with batch norm and Adam
```

---

## Repository Structure

```text
Deep Learning/
├── README.md
├── Week 01 Perceptron Fundamentals and Training Concepts/
│   ├── Module 01  Introduction to Deep Learning and Perceptrons/
│   │   ├── Mini_project.ipynb, Phitron_mini_project.ipynb
│   │   ├── Phitron_non_linear_pattern.ipynb
│   │   ├── Grokking Deep Learning (PDF)
│   │   └── Practice Problem/Solution.md
│   ├── Module 02 How Perceptrons Learn – Intuition, Code, and Edge Cases/
│   │   ├── perceptron_from_scratch.ipynb, Phitron_2_perceptron.ipynb
│   │   ├── notes (PDF; mostly image pages)
│   │   └── Practice Problem/   AND&XOR.ipynb, written answers, boundary plots
│   └── Module 03  Decision Boundaries and Perceptron Learning/
│       ├── Module 03 Notebook.pdf, Module 03 Notebook.png   (image export, no .ipynb)
│       └── Practice Problem/Solution.md, handwritten update photos
├── Week 02 From Perceptron to Multi-Layer Perceptron/
│   ├── Module 04 Learning in Perceptrons From Gradient Descent to Loss Functions/
│   ├── Module 05 From Activation To Loss Function/
│   ├── Module 06 Foundations of Multi-level Layer Perceptrons and Forward Propagation/
│   │   └── notes only (MLP Notation.pdf, Main Idea.pdf) plus Practice Problem/Solution.md
│   └── Module 07 DL Assignment 01/Assignment_01.ipynb
├── Week 03 Backpropagation & Implementaion/
│   ├── Conceptual Session/
│   │   ├── Backprop_Autograd_PyTorch_Pipeline.ipynb
│   │   └── Week 03 CS-02 Backprop, Autograd & PyTorch Training Pipeline/
│   │       └── Phitron_Backprop_Autograd_PyTorch_Pipeline.ipynb
│   ├── Module 08 Backpropagation Fundamentals/   notes, 56-slide deck, written solution
│   ├── Module 09 Pytorch Basic and Backpropagation Implementation/
│   │   ├── Tensors.ipynb, backprop notebooks, Practice Problem/Solution.ipynb
│   └── Module 10 From Computation Graph to Autograd Implementation/
│       └── autograd notebooks; Practice Problem/ has questions only
├── Week 04 Fundamentals of the PyTorch Training Workflow/
│   ├── Conceptual Session/Week 04 CS — PyTorch Training Loop Implementation/
│   │   ├── Hand_Written_Digit_Classification.ipynb
│   │   ├── Phitron_Hand_Written_Digit_Classification.ipynb
│   │   └── train_*.npy, test_*.npy
│   ├── Module 11 Pytorch Training Pipeline From Scratch/   practice questions only
│   ├── Module 12 PyTorch nn Module & torch.optim/
│   ├── Module 13 Training Dynamics in PyTorch – Batch, SGD & Mini-Batch/
│   └── Module 14 DL Mid Term Exam/Midterm_Exam.ipynb
├── Week 05 Project Week - Applied Deep Learning Project/
│   ├── Conceptual Session/ann_hand_mnist_pytorch.ipynb
│   ├── Module 15 Deep Learning Project Setup (Dataset and DataLoader inPyTorch)/
│   ├── Module 16 From ANN Implementation to GPU Training with PyTorch/
│   │   └── fmnist_small.csv plus CPU and GPU notebooks
│   └── Module 17 Loss Functions and Training Challenges in Deep Neural Networks/
├── Week 06 Improving Neural Network Training and Performance/
│   ├── Conceptual Session/
│   │   ├── standard_scaler_pytorch.ipynb
│   │   └── min_max_scaler_pytorch.ipynb
│   ├── Module 18 Data Scaling, Standardization and Normalization/
│   │   └── Social_Network_Ads.csv, scaling notebooks, short notes
│   ├── Module 19 Batch Normalization/
│   │   └── batch-norm notebook, notes, Ioffe & Szegedy paper
│   ├── Module 20 Optimizers/   PDF notes only
│   └── Module 21 DL Final Exam/DL_Final_Exam_Solution.ipynb
└── Optional/Week 01  Perceptron Fundamentals and Training Concepts/
    ├── Perceptron.ipynb
    └── Perceptron Mini Project.ipynb
```

Folder names are kept as they are in the repo, including `Implementaion`, `inPyTorch`, `Multi-level`, and the double spaces in a few Week 01 paths. The en dash in Module 02 and Module 13, and the em dash in the Week 04 conceptual-session folder, are part of the real names.

---

## Topics Covered

| Area | Theory / written work | Implemented in code |
|---|---|---|
| **Perceptron** | Weighted sum, bias, step function, the update rule, linear separability, AND/OR versus XOR, decision-boundary geometry | A NumPy perceptron trained on AND, OR, and XOR; scikit-learn `Perceptron` on Iris (setosa vs. the rest); boundary plots |
| **Loss and gradient descent** | Hinge-style perceptron loss for ±1 labels versus the 0/1 update, why the “plus” in the update is still gradient descent, regression loss | Batch gradient descent for a line (`GDRegressor`) checked against `LinearRegression`; logistic loss written by hand in the assignment |
| **Activations** | Sigmoid as a squashing function, ReLU and vanishing gradients, softmax versus sigmoid, binary cross-entropy versus squared error | Curves and derivatives for sigmoid and ReLU in my notebook; the course notebook also plots tanh, Leaky ReLU, ELU, SELU, softmax, Swish, softplus, hard sigmoid, and softsign |
| **Multi-layer networks** | Notation, forward propagation, why linear layers without an activation collapse into one linear layer | Forward-pass arithmetic in the written solutions and in the later PyTorch models. Module 06 itself has no training notebook |
| **Backpropagation** | A full 3→2→2→2→1 sigmoid network worked by hand, with gradients that shrink toward the input. Chain rule, computation graphs | Manual updates on small PyTorch tensors; Autograd checks; a NumPy XOR network (linear → ReLU → linear → sigmoid) beside `nn.Module` |
| **PyTorch workflow** | What `nn`, `optim`, and a training step are for. Batch, stochastic, and mini-batch gradient descent are compared in the notes and in the midterm write-up | `nn.Module`, `nn.Sequential`, `BCELoss`, `CrossEntropyLoss`, `SGD`, `Adam`, custom `Dataset`, `DataLoader`, `TensorDataset`, `torchinfo.summary`. The runnable batch-size comparison is full-batch versus batch size 1, not a third mini-batch run |
| **Data preparation** | When to standardize and when to min-max scale; fit the scaler on train only; do not scale an ID or a category | `StandardScaler` and `MinMaxScaler`, plus small PyTorch versions of both, on Age and EstimatedSalary |
| **Training difficulties** | Vanishing gradients from repeated sigmoid derivatives; batch normalization (course notes and the Ioffe & Szegedy paper); optimizer families in the Module 20 notes | A 16-layer sigmoid net whose training loss stays near 0.69; `BatchNorm1d` inserted into the Fashion-MNIST ANN. Momentum, AdaGrad, and RMSProp are explained in notes and are not trained in code |
| **Evaluation** | How to read a confusion matrix and a classification report (exam write-ups) | Accuracy, classification report, confusion matrix, and loss curves on the exam and digit notebooks |

---

## Technologies & Tools

| Category | What the notebooks actually use |
|---|---|
| Language | Python. One saved Colab traceback refers to Python 3.12. There is no `requirements.txt`, and no package versions are pinned |
| Deep learning | PyTorch (`torch`, `torch.nn`, `torch.optim`, `Dataset`, `DataLoader`). `torchvision.datasets` downloads Fashion-MNIST and KMNIST in the exams |
| Arrays and tables | NumPy, pandas |
| Classical ML and metrics | scikit-learn: `Perceptron`, `LogisticRegression`, `LinearRegression`, `StandardScaler`, `MinMaxScaler`, `LabelEncoder`, `train_test_split`, `accuracy_score`, `classification_report`, `confusion_matrix`, `make_blobs`, `make_moons`, `make_regression`, `load_iris` |
| Other deep-learning library | TensorFlow / Keras in two places only: the spiral comparison (`Sequential` + `Dense`) and the activation-function plots. The spiral notebook also imports PyTorch and does not train with it |
| Visualization | Matplotlib. Seaborn is imported in the perceptron, AND/XOR, and digit notebooks |
| Extras | `joblib` (save and load the course perceptron), `torchinfo`, `kagglehub` (Week 06 conceptual sessions) |
| Environment | Jupyter notebooks written for Google Colab. Several hard-code `/content/...` paths. GPU notebooks use CUDA when `torch.cuda.is_available()` is true and otherwise fall back to CPU |

---

## Learning Modules

### Week 01 – Perceptron Fundamentals and Training Concepts

**Module 01 – Introduction, and two small classifiers.** The practice solution works a weighted sum by hand (`z = 5` for the given inputs) and defines bias, weights, and the step function. [Mini_project.ipynb](Week%2001%20Perceptron%20Fundamentals%20and%20Training%20Concepts/Module%2001%20%20Introduction%20to%20Deep%20Learning%20and%20Perceptrons/Mini_project.ipynb) trains scikit-learn’s `Perceptron` on Iris after scaling. The label is setosa versus everything else, which is linearly separable, with an 80/20 split (`random_state=42`). The saved train and test accuracy are both 1.0 on 30 test rows. The [course notebook](Week%2001%20Perceptron%20Fundamentals%20and%20Training%20Concepts/Module%2001%20%20Introduction%20to%20Deep%20Learning%20and%20Perceptrons/Phitron_mini_project.ipynb) is the same task with `test_size=0.3`, and its saved report is also perfect, on 45 test rows. A perfect score here means the line separated setosa. It is not a hard problem.

[Phitron_non_linear_pattern.ipynb](Week%2001%20Perceptron%20Fundamentals%20and%20Training%20Concepts/Module%2001%20%20Introduction%20to%20Deep%20Learning%20and%20Perceptrons/Phitron_non_linear_pattern.ipynb) is the contrast. Two noisy spirals, 500 points each, are split 80/20. Logistic regression reaches about 0.71 train and 0.77 test accuracy. A Keras network, `Dense(256, relu) → Dense(256, relu) → Dense(1, sigmoid)`, trained for 50 epochs with Adam and binary cross-entropy, records train and test accuracy 1.0. The spiral coordinates are drawn with unseeded NumPy, so that 1.0 is one saved run. There is no unprefixed copy of this notebook. The module folder also contains a PDF of *Grokking Deep Learning* by Andrew W. Trask.

**Module 02 – How a perceptron learns.** [perceptron_from_scratch.ipynb](Week%2001%20Perceptron%20Fundamentals%20and%20Training%20Concepts/Module%2002%20How%20Perceptrons%20Learn%20%E2%80%93%20Intuition%2C%20Code%2C%20and%20Edge%20Cases/perceptron_from_scratch.ipynb) is my NumPy perceptron: a step activation, an epoch loop, and the update `weight += learning_rate * error * x`. AND and OR settle. XOR is trained for up to 1,000 epochs and the saved weights stop moving without a separating line. The [course notebook](Week%2001%20Perceptron%20Fundamentals%20and%20Training%20Concepts/Module%2002%20How%20Perceptrons%20Learn%20%E2%80%93%20Intuition%2C%20Code%2C%20and%20Edge%20Cases/Phitron_2_perceptron.ipynb) adds the biological-neuron introduction, the sign function, the ±1 derivation, and a `joblib` save/load. Its closing note matches the experiment: AND and OR are linear, XOR is not.

The notes (`intro.pdf`, `weight_updation.pdf`, `solve_and_gate.pdf`) are OneNote exports. Almost no text comes out of them. The written answers are in [Module_02_Practice_Answers.md](Week%2001%20Perceptron%20Fundamentals%20and%20Training%20Concepts/Module%2002%20How%20Perceptrons%20Learn%20%E2%80%93%20Intuition%2C%20Code%2C%20and%20Edge%20Cases/Practice%20Problem/Module_02_Practice_Answers.md): 28 questions on the update rule, bias, `np.dot`, and why XOR needs a hidden layer. [AND&XOR.ipynb](Week%2001%20Perceptron%20Fundamentals%20and%20Training%20Concepts/Module%2002%20How%20Perceptrons%20Learn%20%E2%80%93%20Intuition%2C%20Code%2C%20and%20Edge%20Cases/Practice%20Problem/AND%26XOR.ipynb) draws the AND line `x1 + x2 = 1.5` and the XOR square whose classes sit on crossing diagonals.

**Module 03 – Decision boundaries.** There is no notebook. `Module 03 Notebook.pdf` has no extractable text, and `Module 03 Notebook.png` is the same material as a picture. What I can read is [Solution.md](Week%2001%20Perceptron%20Fundamentals%20and%20Training%20Concepts/Module%2003%20%20Decision%20Boundaries%20and%20Perceptron%20Learning/Practice%20Problem/Solution.md). Weights tilt the boundary, bias slides it, and a worked update takes `w = [0.2, -0.1]`, `b = 0.1`, `η = 0.1`, and a misclassified point `(1, 1)`. The arithmetic is in the two photos next to that file.

### Week 02 – From Perceptron to Multi-Layer Perceptron

**Module 04 – Gradient descent and loss.** The note PDFs are image-heavy. The readable theory is [Solution.md](Week%2002%20From%20Perceptron%20to%20Multi-Layer%20Perceptron/Module%2004%20Learning%20in%20Perceptrons%20From%20Gradient%20Descent%20to%20Loss%20Functions/Practice%20Problem/Solution.md). It derives the ±1 perceptron from hinge loss and shows that `w ← w + η y x` is ordinary gradient descent once the minus sign inside the loss is taken into account. It also points out a real limit of that loss: it goes to zero as soon as a point is barely on the right side, with no margin.

[gradient_descent.ipynb](Week%2002%20From%20Perceptron%20to%20Multi-Layer%20Perceptron/Module%2004%20Learning%20in%20Perceptrons%20From%20Gradient%20Descent%20to%20Loss%20Functions/gradient_descent.ipynb) and the [course copy](Week%2002%20From%20Perceptron%20to%20Multi-Layer%20Perceptron/Module%2004%20Learning%20in%20Perceptrons%20From%20Gradient%20Descent%20to%20Loss%20Functions/Phitron_gradient_descent.ipynb) fit `y = mx + b` by hand on `make_regression` (100 points, one feature, noise 20) and place scikit-learn’s `LinearRegression` next to it. The saved test R² of that sklearn line is 0.635. The notebooks print the moving slope and intercept; they do not report a final R² for the hand-written regressor.

**Module 05 – Activations, then binary cross-entropy.** [3_activation_functions.ipynb](Week%2002%20From%20Perceptron%20to%20Multi-Layer%20Perceptron/Module%2005%20From%20Activation%20To%20Loss%20Function/3_activation_functions.ipynb) plots sigmoid and ReLU and a numerical derivative. [Phitron_3_activation_functions.ipynb](Week%2002%20From%20Perceptron%20to%20Multi-Layer%20Perceptron/Module%2005%20From%20Activation%20To%20Loss%20Function/Phitron_3_activation_functions.ipynb) is the long version: tanh, Leaky ReLU, ELU, SELU, softmax, Swish, softplus, hard sigmoid, and softsign, with formulas in the markdown and TensorFlow used for several of the reference implementations. The practice write-up computes `σ(0) = 0.5`, `σ(2) ≈ 0.88`, `σ(-2) ≈ 0.12`, and two binary-cross-entropy examples (about 0.05 for a confident correct guess, about 0.70 for a confident wrong one).

**Module 06 – MLP notation and forward propagation.** Notes only, plus [Solution.md](Week%2002%20From%20Perceptron%20to%20Multi-Layer%20Perceptron/Module%2006%20Foundations%20of%20Multi-level%20Layer%20Perceptrons%20and%20Forward%20Propagation/Practice%20Problem/Solution.md). The point I keep coming back to is the collapse: `W2(W1 x + b1) + b2` is still one affine map. Depth starts to matter when a non-linear activation sits between the layers. The same file spells out input, hidden, and output layers, and forward propagation as weighted sum, activation, and pass-it-on.

**Module 07 – Assignment 01, NeuroNav.** [Assignment_01.ipynb](Week%2002%20From%20Perceptron%20to%20Multi-Layer%20Perceptron/Module%2007%20DL%20Assignment%2001/Assignment_01.ipynb) is a story about Forest versus Urban, but the data is `make_blobs` (200 points, 2 features, 2 centers, `random_state=42`), not satellite imagery. I implemented sigmoid, binary cross-entropy, a forward pass, and gradient descent for 500 epochs at learning rate 0.5, then plotted the decision boundary. The notebook’s own tests for questions 2 through 5 are saved as passed. The short written answers say why a sigmoid output suits a two-class problem, and why a single perceptron still fails if those two classes are not linearly separable. No accuracy number is stored.

### Week 03 – Backpropagation and Implementation

The week folder is spelled `Implementaion` in the repository.

**Module 08 – Backpropagation by hand.** `Backpropagation.pdf` yields only a title. The slide deck has 56 slides. The calculation I can actually read is [Backpropagation_Solution.md](Week%2003%20Backpropagation%20%26%20Implementaion/Module%2008%20Backpropagation%20Fundamentals/Practice%20Problem/Backpropagation_Solution.md): a 3→2→2→2→1 network, sigmoid everywhere, squared error, one example `x = [1, 2, 3]`, target 0.8, learning rate 0.1. The forward pass gives `ŷ ≈ 0.560` and loss 0.0578. After one gradient step the loss is about 0.0556. The same write-up is where vanishing gradients show up as numbers: output-layer gradients are on the order of `10⁻¹` to `10⁻²`, and the first-layer gradients have fallen to about `10⁻⁵`, because each sigmoid derivative is at most 0.25.

**Module 09 – Tensors, then a hand-updated network.** [Tensors.ipynb](Week%2003%20Backpropagation%20%26%20Implementaion/Module%2009%20Pytorch%20Basic%20and%20Backpropagation%20Implementation/Tensors.ipynb) is my long tour of creation, dtypes, broadcasting, reductions, and matrix ops. The saved CUDA check says the device was not available. [backprop_using_pytorch.ipynb](Week%2003%20Backpropagation%20%26%20Implementaion/Module%2009%20Pytorch%20Basic%20and%20Backpropagation%20Implementation/backprop_using_pytorch.ipynb) and the [course copy](Week%2003%20Backpropagation%20%26%20Implementaion/Module%2009%20Pytorch%20Basic%20and%20Backpropagation%20Implementation/Phitron_backprop_using_pytorch.ipynb) build a two-layer forward pass and then poke individual weights with a hand-written error signal for 5 epochs. Saved loss falls from 14.68 to 9.37. That is a teaching loop, not a finished model. [Solution.ipynb](Week%2003%20Backpropagation%20%26%20Implementaion/Module%2009%20Pytorch%20Basic%20and%20Backpropagation%20Implementation/Practice%20Problem/Solution.ipynb) checks shapes (`X` is `1×2`, `W` is `2×1`), a ReLU forward pass, and squared error `(5 − 3)² = 4`.

**Module 10 – Computation graphs and Autograd.** [pytorch_autograd.ipynb](Week%2003%20Backpropagation%20%26%20Implementaion/Module%2010%20From%20Computation%20Graph%20to%20Autograd%20Implementation/pytorch_autograd.ipynb) and the [course notebook](Week%2003%20Backpropagation%20%26%20Implementaion/Module%2010%20From%20Computation%20Graph%20to%20Autograd%20Implementation/Phitron_pytorch_autograd.ipynb) differentiate small formulas by hand and with `backward()`, then do the same for a sigmoid binary-cross-entropy term. Saved outputs include the failures that belong to the lesson: `RuntimeError` when a tensor does not require grad. The practice PDF asks what a computation graph is and why Autograd exists. I did not put a solution next to it.

**Conceptual session – backprop, Autograd, and a first training loop.** My notebook is [Backprop_Autograd_PyTorch_Pipeline.ipynb](Week%2003%20Backpropagation%20%26%20Implementaion/Conceptual%20Session/Backprop_Autograd_PyTorch_Pipeline.ipynb), one level above the course copy in the `Week 03 CS-02 ...` folder. Both use a 4×2 XOR toy set and a 2→4→1 network. Part 1 is NumPy (ReLU hidden layer, sigmoid output, binary cross-entropy, learning rate 0.5). On my saved run the loss moves from 0.711 to 0.477 over 4,000 epochs and the predictions stay near 0.67, so the hand-written net has not solved XOR. The course notebook’s saved NumPy run sits on 0.693 the whole way. Part 3 rebuilds the same shape as `TinyNet` (`nn.Linear` → ReLU → `nn.Linear` → sigmoid) with `BCELoss` and SGD. That version does learn: my saved loss goes from 0.704 to 0.0009, and the course notebook’s saved run goes from 0.713 to 0.0009. Same architecture on paper, very different saved outcomes between the manual loop and `nn.Module`.

### Week 04 – Fundamentals of the PyTorch Training Workflow

**Module 11 – A training pipeline without `nn.Module`.** [pytorch_training_pipeline.ipynb](Week%2004%20Fundamentals%20of%20the%20PyTorch%20Training%20Workflow/Module%2011%20Pytorch%20Training%20Pipeline%20From%20Scratch/pytorch_training_pipeline.ipynb) downloads the Breast Cancer Wisconsin table from GitHub, drops `id` and the empty `Unnamed: 32` column, encodes malignant/benign, and scales the 30 features. The model is one weight vector, a bias, a sigmoid, and binary cross-entropy, updated with Autograd for 25 full-batch epochs at learning rate 0.1. The saved test accuracy is **0.622**. The cutoff in that cell is `y_pred > 0.9`, not 0.5, and `train_test_split` has no `random_state`, so the number is one unseeded split scored with a harsh threshold. The [course notebook](Week%2004%20Fundamentals%20of%20the%20PyTorch%20Training%20Workflow/Module%2011%20Pytorch%20Training%20Pipeline%20From%20Scratch/Phitron_pytorch_training_pipeline.ipynb) records loss falling from 3.87 to 0.75 and test accuracy **0.638** under the same kind of setup. The practice PDF (a `requires_grad` question) has no solution file.

**Module 12 – `nn.Module` and `torch.optim`.** [nn_module.ipynb](Week%2004%20Fundamentals%20of%20the%20PyTorch%20Training%20Workflow/Module%2012%20PyTorch%20nn%20Module%20%26%20torch.optim/nn_module.ipynb) builds the same tiny stack three ways and prints a `torchinfo` summary: `Linear(features, 3) → ReLU → Linear(3, 1) → Sigmoid`. The course copy saves an `AttributeError` from asking a `Sigmoid` module for `.weight`, which is a useful reminder that not every submodule has parameters.

[pytorch_training_pipeline_updated.ipynb](Week%2004%20Fundamentals%20of%20the%20PyTorch%20Training%20Workflow/Module%2012%20PyTorch%20nn%20Module%20%26%20torch.optim/pytorch_training_pipeline_updated.ipynb) puts the Breast Cancer model into `nn.Linear(num_features, 1)` plus `Sigmoid`, with `BCELoss` and `SGD` (learning rate 0.1, 25 epochs). Saved test accuracy, again at a 0.9 cutoff and an unseeded split: **0.542** in my notebook and **0.596** in the [course notebook](Week%2004%20Fundamentals%20of%20the%20PyTorch%20Training%20Workflow/Module%2012%20PyTorch%20nn%20Module%20%26%20torch.optim/Phitron_pytorch_training_pipeline_updated.ipynb). I would not rank these against the 0.622 from Module 11. The split changes every run, and the cutoff throws away predictions that a 0.5 rule would have counted.

**Module 13 – Batch versus stochastic updates.** The notes and [Extra Resources.docx](Week%2004%20Fundamentals%20of%20the%20PyTorch%20Training%20Workflow/Module%2013%20Training%20Dynamics%20in%20PyTorch%20%E2%80%93%20Batch%2C%20SGD%20%26%20Mini-Batch/Extra%20Resources.docx) describe three variants: full batch, one example, and mini-batch. The code only runs two of them. [batch_vs_stochastic.ipynb](Week%2004%20Fundamentals%20of%20the%20PyTorch%20Training%20Workflow/Module%2013%20Training%20Dynamics%20in%20PyTorch%20%E2%80%93%20Batch%2C%20SGD%20%26%20Mini-Batch/batch_vs_stochastic.ipynb) builds 50 synthetic points in 2D, a `Linear(2, 1)` with `BCELoss` and SGD at learning rate 0.1, and trains 5 epochs twice: `batch_size=1` and `batch_size=50`. Because the dataset has 50 rows, the second run is full-batch gradient descent. The saved figure is the loss comparison. There is no separate mini-batch curve in the notebook. The markdown under the plot says SGD takes more steps per epoch, so it moves faster when the epoch budget is small, while full-batch loss is smoother.

**Conceptual session – handwritten digits.** The arrays live next to the notebooks: `train_images.npy` is `(60000, 28, 28)` `uint8`, `test_images.npy` is `(10000, 28, 28)`, labels are 0–9, and pixel values run from 0 to 255. [Hand_Written_Digit_Classification.ipynb](Week%2004%20Fundamentals%20of%20the%20PyTorch%20Training%20Workflow/Conceptual%20Session/Week%2004%20CS%20%E2%80%94%20PyTorch%20Training%20Loop%20Implementation/Hand_Written_Digit_Classification.ipynb) loads them from `/content/`, holds out 10% of the training images for validation, and trains `Flatten → Linear(784, 128) → ReLU → Linear(128, 64) → ReLU → Linear(64, 10)` with `CrossEntropyLoss` and SGD at learning rate 0.01 for 10 epochs. Pixels are not divided by 255. The saved losses stay in the hundreds or thousands, and the test accuracy is **0.2870** (2,870 of 10,000). A confusion matrix is saved with that run. The [course notebook](Week%2004%20Fundamentals%20of%20the%20PyTorch%20Training%20Workflow/Conceptual%20Session/Week%2004%20CS%20%E2%80%94%20PyTorch%20Training%20Loop%20Implementation/Phitron_Hand_Written_Digit_Classification.ipynb) trains the same model and leaves accuracy as a comment (“HW”); it has loss curves and no accuracy figure.

**Module 14 – Midterm.** See [Evaluations / Exams](#evaluations--exams).

### Week 05 – Project Week

**Module 15 – `Dataset` and `DataLoader`.** [dataset_and_dataloader_demo.ipynb](Week%2005%20Project%20Week%20-%20Applied%20Deep%20Learning%20Project/Module%2015%20Deep%20Learning%20Project%20Setup%20%28Dataset%20and%20DataLoader%20inPyTorch%29/dataset_and_dataloader_demo.ipynb) is the small version: a `CustomDataset` over a toy matrix and a loader with batch size 4. [pytorch_training_pipeline_with_dataset_dataloader_class.ipynb](Week%2005%20Project%20Week%20-%20Applied%20Deep%20Learning%20Project/Module%2015%20Deep%20Learning%20Project%20Setup%20%28Dataset%20and%20DataLoader%20inPyTorch%29/pytorch_training_pipeline_with_dataset_dataloader_class.ipynb) rebuilds the Breast Cancer logistic regression on top of that, batch size 32, and also shows an index-slicing loop that does not use a loader. The saved score in the “old” evaluation cell is **0.530**, still with the 0.9 cutoff. The cells under “Better version” only print shapes. They do not record a new accuracy. `Week's outline.pdf` is a one-page schedule (this module, then the ANN, then loss and vanishing gradients). `Blog links.docx` points at the PyTorch data-tutorial and a few explainers.

**Module 16 – Fashion-MNIST ANN, CPU and GPU.** Every notebook here reads `/content/fmnist_small.csv`. The file in the repo has **6,000 rows** and 785 columns (`label` plus `pixel1`…`pixel784`). The network is not a CNN. It is `Linear(784, 128) → ReLU → Linear(128, 64) → ReLU → Linear(64, 10)`, `CrossEntropyLoss`, SGD at learning rate 0.1, batch size 32, pixels divided by 255, and an 80/20 split with `random_state=42` (4,800 / 1,200).

| Notebook | Epochs | What was saved |
|---|---|---|
| [ann_fashion_mnist_pytorch_cpu.ipynb](Week%2005%20Project%20Week%20-%20Applied%20Deep%20Learning%20Project/Module%2016%20From%20ANN%20Implementation%20to%20GPU%20Training%20with%20PyTorch/ann_fashion_mnist_pytorch_cpu.ipynb) | 10 | One training-loss number, 0.398, because the print sits outside the epoch loop. Holdout accuracy **0.83** (996/1,200) |
| [Phitron_ann_project_with_cpu.ipynb](Week%2005%20Project%20Week%20-%20Applied%20Deep%20Learning%20Project/Module%2016%20From%20ANN%20Implementation%20to%20GPU%20Training%20with%20PyTorch/Phitron_ann_project_with_cpu.ipynb) | 10 | Final-epoch training loss 0.079. Holdout accuracy **0.827** (992/1,200) |
| [ann_fashion_mnist_pytorch_gpu.ipynb](Week%2005%20Project%20Week%20-%20Applied%20Deep%20Learning%20Project/Module%2016%20From%20ANN%20Implementation%20to%20GPU%20Training%20with%20PyTorch/ann_fashion_mnist_pytorch_gpu.ipynb) | 100 | Device recorded as `cuda`. Training loss falls to 0.003. Training accuracy **0.99875** (4,794/4,800). Holdout accuracy **0.824** (989/1,200) |
| [Phitron_ann_fashion_mnist_pytorch_gpu.ipynb](Week%2005%20Project%20Week%20-%20Applied%20Deep%20Learning%20Project/Module%2016%20From%20ANN%20Implementation%20to%20GPU%20Training%20with%20PyTorch/Phitron_ann_fashion_mnist_pytorch_gpu.ipynb) | 100 | Device recorded as `cuda`. The saved holdout accuracy is **0.8869** and the training accuracy is **0.9796**. Those fractions match a 12,000-image holdout and a 48,000-image training set, not the 6,000-row CSV in the repo |

The 100-epoch GPU run on the file that is actually here reaches a slightly lower holdout score than the 10-epoch CPU runs, with training accuracy essentially perfect. That is overfitting, not evidence that the GPU is worse. The course GPU notebook’s saved scores cannot be checked against `fmnist_small.csv` as it stands.

**Module 17 – A deep net that does not learn.** [Phitron_vanishing_gradient_problem.ipynb](Week%2005%20Project%20Week%20-%20Applied%20Deep%20Learning%20Project/Module%2017%20Loss%20Functions%20and%20Training%20Challenges%20in%20Deep%20Neural%20Networks/Phitron_vanishing_gradient_problem.ipynb) builds `make_moons` (250 points, noise 0.05) and a sigmoid network with fifteen hidden layers (2→10, then fourteen layers of 10→10) plus a sigmoid output. `BCELoss`, Adam, learning rate 0.001, 100 epochs, trained on the 200-point training slice only. Saved loss goes from 0.6949 to 0.6929. It never leaves the “guess 50/50” region. The notebook then plots how far the **first-layer** weights moved, and it prints `(initial − final) / 0.001` under the name “Gradient”. That quantity is the whole training change divided by 0.001. It is not a single backward pass, and it is not a layer-by-layer gradient table. The place that actually multiplies sigmoid derivatives backward, layer by layer, is the Module 08 written solution.

[vanishing_gradient_problem.ipynb](Week%2005%20Project%20Week%20-%20Applied%20Deep%20Learning%20Project/Module%2017%20Loss%20Functions%20and%20Training%20Challenges%20in%20Deep%20Neural%20Networks/vanishing_gradient_problem.ipynb) still has the class name `SigmoidNet`, but the sigmoid stack is commented out and a ReLU stack is what the source would run. The saved loss and first-layer numbers match the sigmoid notebook almost digit for digit. I treat those outputs as the sigmoid run, not as a ReLU result.

**Conceptual session – Sign Language MNIST.** [ann_hand_mnist_pytorch.ipynb](Week%2005%20Project%20Week%20-%20Applied%20Deep%20Learning%20Project/Conceptual%20Session/ann_hand_mnist_pytorch.ipynb) expects `sign_mnist_train.csv` and `sign_mnist_test.csv` in the working directory. Those files are not in the repository. The notebook’s saved output shows the usual gap: raw labels skip 9, because J (and Z) need motion. It remaps the remaining letters to 24 classes (`A`–`Y` except `J`), keeps 100 images per letter (2,400 total), and splits 80/20 stratified (1,920 / 480). Pixels are divided by 255. The model is again fully connected, `784 → 128 → 64 → 24`, SGD at learning rate 0.1, batch size 32, 20 epochs. Saved training loss falls from 3.18 to about 1.66. Saved accuracy is **0.577** on the training subset and **0.529** on the 480-image holdout. The notebook says this is a still photo of a hand, not fingerspelling.

### Week 06 – Improving Neural Network Training and Performance

**Conceptual session – scaling Age and EstimatedSalary.** Both notebooks download Social Network Ads with `kagglehub` (`rakeshrau/social-network-ads`). The same table is also stored at [Social_Network_Ads.csv](Week%2006%20Improving%20Neural%20Network%20Training%20and%20Performance/Module%2018%20Data%20Scaling%2C%20Standardization%20and%20Normalization/Social_Network_Ads.csv) (400 rows: `User ID`, `Gender`, `Age`, `EstimatedSalary`, `Purchased`). The lesson keeps Age and EstimatedSalary only. User ID and Gender are left out on purpose.

The model is logistic regression: `nn.Linear(2, 1)` and `BCEWithLogitsLoss`, SGD, learning rate 0.1, 200 epochs, an 80/20 split (320 / 80). The split uses `torch.randperm` and is not seeded. Model initialization inside the training function is seeded. Both notebooks record the same unscaled run: loss starts at 22,180 and climbs past 10⁸, with train accuracy 0.634 and **test accuracy 0.675**. After scaling, the saved test accuracy is **0.800** with the [standard scaler](Week%2006%20Improving%20Neural%20Network%20Training%20and%20Performance/Conceptual%20Session/standard_scaler_pytorch.ipynb) (train 0.859) and **0.762** with the [min-max scaler](Week%2006%20Improving%20Neural%20Network%20Training%20and%20Performance/Conceptual%20Session/min_max_scaler_pytorch.ipynb) (train 0.756). Those three test numbers are the same experiment, not three different datasets. They are one unseeded split. They do not say that standardization is universally better than min-max.

**Module 18 – the same table, a wider network.** [data_scaling_example_pytorch.ipynb](Week%2006%20Improving%20Neural%20Network%20Training%20and%20Performance/Module%2018%20Data%20Scaling%2C%20Standardization%20and%20Normalization/data_scaling_example_pytorch.ipynb) reads `/content/Social_Network_Ads.csv`, drops the first two columns, and trains `Linear(2, 128) → ReLU → Linear(128, 1) → Sigmoid` with `BCELoss` and Adam at learning rate 0.001 for 100 epochs. The split is 80/20 with `random_state=2`. Without scaling, saved loss stays at 65.3 and the printed holdout accuracy stays at **0.400** (the notebook labels this “Val Acc”; it is the 80-row holdout, and it is checked during training). With `StandardScaler`, the printed holdout accuracy at epoch 90 is **0.825** in my notebook and **0.8375** in the [course notebook](Week%2006%20Improving%20Neural%20Network%20Training%20and%20Performance/Module%2018%20Data%20Scaling%2C%20Standardization%20and%20Normalization/Phitron_data_scaling_example_pytorch.ipynb). [Standardization.ipynb](Week%2006%20Improving%20Neural%20Network%20Training%20and%20Performance/Module%2018%20Data%20Scaling%2C%20Standardization%20and%20Normalization/Standardization.ipynb) and [Normalization.ipynb](Week%2006%20Improving%20Neural%20Network%20Training%20and%20Performance/Module%2018%20Data%20Scaling%2C%20Standardization%20and%20Normalization/Normalization.ipynb) are the tiny formula checks, `(x − mean) / std` and `(x − min) / (max − min)`, next to sklearn. The note that does extract cleanly, `Standardization vs Normalization.pdf`, says to prefer standardization for neural nets and logistic regression, and min-max when you need a fixed range.

**Module 19 – Batch normalization.** The notes define batch norm as normalizing a layer’s activations with the current mini-batch mean and variance. The folder includes the Ioffe and Szegedy paper, *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift* (arXiv:1502.03167). [ann_fashion_mnist_pytorch_gpu_optimized_with_batchNorm.ipynb](Week%2006%20Improving%20Neural%20Network%20Training%20and%20Performance/Module%2019%20Batch%20Normalization/ann_fashion_mnist_pytorch_gpu_optimized_with_batchNorm.ipynb) inserts `BatchNorm1d` after each hidden linear layer of the Module 16 network and trains 100 epochs on the same 6,000-row CSV. The optimizer that actually runs is SGD with learning rate 0.1 and `weight_decay=1e-4`. A variable `learning_rate = 0.01` is set and never passed in. The device line saved `cuda`. Training loss ends at 0.0037. The saved holdout accuracy is **0.8375** (1,005/1,200). Training accuracy is not recorded. I would not call this “batch norm wins”: the non-batch-norm GPU run used a different optimizer setup, and both runs drive training loss near zero.

**Module 20 – Optimizers.** PDF notes only, no notebook. `optimizers_types.pdf` is the page with extractable text. It walks through batch, stochastic, and mini-batch gradient descent, then momentum, AdaGrad (and the shrinking-step problem), RMSProp, and Adam as momentum plus RMSProp. `EWMA.pdf`, `All Optimizers.pdf`, and `Previous optimizers.pdf` are image exports. `optimizers_pytorch.pdf` has no extractable text.

**Module 21 – Final exam.** See [Evaluations / Exams](#evaluations--exams).

### Optional

[Perceptron.ipynb](Optional/Week%2001%20%20Perceptron%20Fundamentals%20and%20Training%20Concepts/Perceptron.ipynb) is another from-scratch perceptron, trained on AND and OR for 10 epochs at learning rate 0.1. [Perceptron Mini Project.ipynb](Optional/Week%2001%20%20Perceptron%20Fundamentals%20and%20Training%20Concepts/Perceptron%20Mini%20Project.ipynb) repeats the setosa-versus-rest Iris project (`test_size=0.2`, `eta0=0.1`, `max_iter=100`) and again records train and test accuracy 1.0.

---

## Practical Work & Experiments

| Work | Task / dataset | Model or method | Main learning | Link |
|---|---|---|---|---|
| Perceptron from scratch | AND, OR, XOR truth tables | NumPy step-function perceptron | AND and OR converge; XOR does not | [notebook](Week%2001%20Perceptron%20Fundamentals%20and%20Training%20Concepts/Module%2002%20How%20Perceptrons%20Learn%20%E2%80%93%20Intuition%2C%20Code%2C%20and%20Edge%20Cases/perceptron_from_scratch.ipynb) |
| Iris perceptron | Setosa vs. the other two species; features scaled | sklearn `Perceptron` | A linearly separable subset is an easy win. Saved test accuracy 1.0 on 30 rows (mine) and 45 rows (course) | [mine](Week%2001%20Perceptron%20Fundamentals%20and%20Training%20Concepts/Module%2001%20%20Introduction%20to%20Deep%20Learning%20and%20Perceptrons/Mini_project.ipynb) |
| Spiral comparison | Two spirals, 1,000 points, 80/20 | Logistic regression vs. Keras MLP (256, 256, sigmoid) | Linear test accuracy 0.765; the MLP records 1.0. Spiral noise is unseeded | [notebook](Week%2001%20Perceptron%20Fundamentals%20and%20Training%20Concepts/Module%2001%20%20Introduction%20to%20Deep%20Learning%20and%20Perceptrons/Phitron_non_linear_pattern.ipynb) |
| Gradient descent for a line | `make_regression`, 100×1, noise 20 | Hand-written GD vs. `LinearRegression` | Sklearn test R² 0.635. The hand-written fit prints `m` and `b` as they move | [notebook](Week%2002%20From%20Perceptron%20to%20Multi-Layer%20Perceptron/Module%2004%20Learning%20in%20Perceptrons%20From%20Gradient%20Descent%20to%20Loss%20Functions/gradient_descent.ipynb) |
| NeuroNav assignment | `make_blobs`, 200 points, 2 classes | Manual sigmoid, BCE, GD, 500 epochs, lr 0.5 | A full training loop without a framework. Built-in tests passed. No accuracy saved | [notebook](Week%2002%20From%20Perceptron%20to%20Multi-Layer%20Perceptron/Module%2007%20DL%20Assignment%2001/Assignment_01.ipynb) |
| XOR, NumPy then PyTorch | 4-point XOR | 2→4→1, ReLU then sigmoid, BCE | My NumPy loss ends at 0.477. The `nn.Module` run ends at 0.0009 | [notebook](Week%2003%20Backpropagation%20%26%20Implementaion/Conceptual%20Session/Backprop_Autograd_PyTorch_Pipeline.ipynb) |
| Breast Cancer logistic regression | Diagnostic features, 569 rows, downloaded at runtime | One linear layer, sigmoid, BCE, 25 epochs, threshold 0.9 | Saved test accuracy 0.622 from scratch, 0.542 and 0.596 after the `nn.Module` rewrite, 0.530 with a loader. Unseeded splits | [from scratch](Week%2004%20Fundamentals%20of%20the%20PyTorch%20Training%20Workflow/Module%2011%20Pytorch%20Training%20Pipeline%20From%20Scratch/pytorch_training_pipeline.ipynb) |
| Batch vs. SGD | 50 synthetic 2D points | `Linear(2, 1)`, 5 epochs, batch 1 vs. batch 50 | More updates per epoch versus a smoother full-batch curve. Mini-batch is notes-only | [notebook](Week%2004%20Fundamentals%20of%20the%20PyTorch%20Training%20Workflow/Module%2013%20Training%20Dynamics%20in%20PyTorch%20%E2%80%93%20Batch%2C%20SGD%20%26%20Mini-Batch/batch_vs_stochastic.ipynb) |
| Digit ANN | Included `.npy` digits, 60k / 10k, pixels left in 0–255 | 784→128→64→10, SGD, 10 epochs | Test accuracy **0.287**. The scale of the inputs shows up as huge loss | [notebook](Week%2004%20Fundamentals%20of%20the%20PyTorch%20Training%20Workflow/Conceptual%20Session/Week%2004%20CS%20%E2%80%94%20PyTorch%20Training%20Loop%20Implementation/Hand_Written_Digit_Classification.ipynb) |
| Fashion-MNIST ANN | `fmnist_small.csv`, 6,000 rows, pixels / 255 | Same 784→128→64→10 MLP | 10-epoch CPU holdout about 0.83. 100-epoch GPU holdout 0.824 with training accuracy 0.999 | [GPU notebook](Week%2005%20Project%20Week%20-%20Applied%20Deep%20Learning%20Project/Module%2016%20From%20ANN%20Implementation%20to%20GPU%20Training%20with%20PyTorch/ann_fashion_mnist_pytorch_gpu.ipynb) |
| Sign-language ANN | 24 letters, 100 images each, CSVs not in the repo | 784→128→64→24, 20 epochs | Holdout accuracy 0.529 on 480 images. J and Z are absent because they move | [notebook](Week%2005%20Project%20Week%20-%20Applied%20Deep%20Learning%20Project/Conceptual%20Session/ann_hand_mnist_pytorch.ipynb) |
| Deep sigmoid net | `make_moons`, 250 points | 15 sigmoid hidden layers, Adam, 100 epochs | Loss stays at 0.693. The printed “gradient” is a rescaled weight change, not a backprop dump | [notebook](Week%2005%20Project%20Week%20-%20Applied%20Deep%20Learning%20Project/Module%2017%20Loss%20Functions%20and%20Training%20Challenges%20in%20Deep%20Neural%20Networks/Phitron_vanishing_gradient_problem.ipynb) |
| Social Network Ads, linear | Age and salary, 400 rows, 80/20 | `Linear(2, 1)`, `BCEWithLogitsLoss`, 200 epochs | Test accuracy 0.675 unscaled, 0.800 standardized, 0.762 min-max. Unseeded split | [standard](Week%2006%20Improving%20Neural%20Network%20Training%20and%20Performance/Conceptual%20Session/standard_scaler_pytorch.ipynb) · [min-max](Week%2006%20Improving%20Neural%20Network%20Training%20and%20Performance/Conceptual%20Session/min_max_scaler_pytorch.ipynb) |
| Social Network Ads, wider net | Same two features, seeded 80/20 | 2→128→1, Adam, 100 epochs | Holdout accuracy stuck at 0.400 unscaled; 0.825 at epoch 90 after `StandardScaler` | [notebook](Week%2006%20Improving%20Neural%20Network%20Training%20and%20Performance/Module%2018%20Data%20Scaling%2C%20Standardization%20and%20Normalization/data_scaling_example_pytorch.ipynb) |
| Batch norm on the small Fashion CSV | Same 6,000-row file and 80/20 split | MLP plus `BatchNorm1d`, SGD lr 0.1, weight decay 1e-4, 100 epochs | Holdout accuracy 0.8375. Training loss near zero. Not a controlled comparison with Module 16 | [notebook](Week%2006%20Improving%20Neural%20Network%20Training%20and%20Performance/Module%2019%20Batch%20Normalization/ann_fashion_mnist_pytorch_gpu_optimized_with_batchNorm.ipynb) |

---

## Evaluations / Exams

Both exams live in a single notebook each. The questions and my answers are in the same file. Nothing in the repository records a mark, so I am not inventing one. The percentages below are model accuracy on the official test split, not an exam grade.

### Midterm – StyleAI (Week 04, Module 14)

[Midterm_Exam.ipynb](Week%2004%20Fundamentals%20of%20the%20PyTorch%20Training%20Workflow/Module%2014%20DL%20Mid%20Term%20Exam/Midterm_Exam.ipynb)

The story is a wardrobe app that names a clothing photo. The data is Fashion-MNIST from `torchvision` (not the small CSV from Week 05): 60,000 training images and 10,000 test images, 28×28 grayscale, ten classes from T-shirt/top through Ankle boot.

- **Question 1 (10 marks)** is written: sigmoid versus ReLU, the three gradient-descent variants, why `CrossEntropyLoss` is the multi-class loss, why the network should not end in an extra `Softmax` when that loss already applies it, and what `loss.backward()` does.
- **Data.** Pixels are divided by 255. One sixth of the 60,000 training images is held out as validation. The reported test number uses the official 10,000-image test set. Loaders use batch size 64.
- **Model.** `FashionClassifier` is `Flatten → Linear(784, 256) → ReLU → Linear(256, 128) → ReLU → Linear(128, 64) → ReLU → Linear(64, 10)`. No convolution.
- **Training.** `CrossEntropyLoss`, SGD, learning rate 0.01, 15 epochs. Saved losses: epoch 1 is 2.1632 train / 1.6019 validation; epoch 15 is **0.4078 train / 0.4721 validation**.
- **Test accuracy: 82.45%** (the print in the notebook). The classification report rounds the same run to 0.82 on 10,000 images. Per class, Shirt is the weak one in that report: precision 0.63, recall 0.46, F1 0.53. Coat has recall 0.90 and precision 0.59. Trouser, Sandal, Sneaker, Bag, and Ankle boot are all above 0.90 F1.
- **Plots.** Training and validation loss, training and validation accuracy, and a confusion matrix are saved as figures.
- **Written note versus the report.** The note says Shirt precision 0.57 and recall 0.67, and that Shirt is confused with T-shirt/top. The saved classification report does not match those two figures. It does show Shirt as the lowest-F1 class. The note also calls the curves slightly overfitting, which fits the small gap between the epoch-15 losses above. The pairwise counts live in the confusion-matrix image; they are not printed as a table.

### Final exam – TranslateAI (Week 06, Module 21)

[DL_Final_Exam_Solution.ipynb](Week%2006%20Improving%20Neural%20Network%20Training%20and%20Performance/Module%2021%20DL%20Final%20Exam/DL_Final_Exam_Solution.ipynb)

The story is ancient-character recognition. The data is KMNIST from `torchvision`: 28×28 grayscale images of ten cursive characters, `o, ki, su, tsu, na, ha, ma, ya, re, wo`.

- **Data.** Same pattern as the midterm: divide by 255, hold out one sixth of the training images for validation, test on the official test set. Loaders use batch size 256, shuffle on for training only.
- **Model.** `TranslateAIClassifier` is fully connected, with batch norm on the hidden widths: `784 → 512 → 256 → 128 → 10`, `BatchNorm1d` and ReLU after each hidden linear layer, then a linear map to 10 classes. No convolution.
- **Training.** `CrossEntropyLoss`, Adam, learning rate 0.001, 15 epochs. Saved losses: epoch 1 is 0.380 train / 0.195 validation; epoch 15 is **0.0097 train / 0.144 validation**. Training loss falls much faster than validation loss.
- **Test accuracy: 91.86%.** A confusion matrix is saved. There is no classification report in the outputs.
- **Written note.** One sentence: the main confusion is `ha` read as `su`. The notebook does not print the cell counts behind that sentence.

---

## Skills Developed & Learning Outcomes

What the files show I can do:

- Implement a perceptron, explain when a straight decision boundary is enough, and point at XOR when it is not.
- Derive the perceptron update from a loss, and run gradient descent on a one-dimensional regression by hand.
- Pick an activation and a loss for the task in front of me: step or sigmoid for a hard or soft binary decision, ReLU in hidden layers, softmax with cross-entropy only when there are several exclusive classes.
- Finish a numeric backward pass, and recognize vanishing gradients both in that arithmetic and in a deep sigmoid net whose loss never leaves 0.69.
- Use PyTorch Autograd, and see the error that appears when a tensor is not tracked.
- Write the training step both as explicit tensors and as `nn.Module` + `torch.optim`.
- Build a `Dataset` and a `DataLoader`, including a flattened-image pipeline, and move that loop to GPU when CUDA is present.
- Scale features, fit the scaler on the training split only, and show a run where unscaled Age and Salary make the loss explode.
- Add `BatchNorm1d`, and read the Module 20 notes on momentum, AdaGrad, RMSProp, and Adam even though only SGD and Adam appear in training code.
- Report accuracy together with a confusion matrix or a classification report, and notice when a saved number is a tiny split, an unseeded split, a harsh threshold, or a training-set score.

---

## How to Use

### Suggested order

Read the weeks in order. Inside a week, follow the module numbers. When a `Phitron_` notebook and an unprefixed notebook sit side by side, the prefixed file is the class version and the other one is my run. For Week 03, my conceptual-session notebook is directly under `Conceptual Session/`, and the course version is inside the `Week 03 CS-02 ...` folder.

Module 06, Module 20, and most of the OneNote PDFs are reading, not code. Module 10 and Module 11 have practice questions and no solution file.

### Google Colab

Most notebooks were executed in Colab. Paths you will have to fix, or files you will have to upload, include:

- `/content/train_images.npy`, `train_labels.npy`, `test_images.npy`, `test_labels.npy` in the Week 04 conceptual session. The arrays are in that folder in the repo.
- `/content/fmnist_small.csv` in Module 16 and Module 19. The CSV is in the Module 16 folder.
- `/content/Social_Network_Ads.csv` in Module 18. The CSV is in the Module 18 folder. The Week 06 conceptual sessions ignore that path and call `kagglehub` instead, which needs Kaggle access.
- `sign_mnist_train.csv` and `sign_mnist_test.csv` for the Week 05 conceptual session. They are not in the repo.

The exam notebooks call `torchvision.datasets.FashionMNIST` and `KMNIST` with `download=True`, so they fetch the data on first run. The Breast Cancer notebooks read a GitHub raw URL. Iris and the `make_*` datasets need no download.

GPU notebooks check `torch.cuda.is_available()`. The Fashion-MNIST GPU notebook and the batch-norm notebook saved `Using device: cuda`. The tensor tour saved “device is not available”. On CPU, the 100-epoch image runs are the slow ones.

### Running locally

```bash
git clone https://github.com/ak1bhasan/Deep-Learning.git
cd Deep-Learning

pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn jupyter
pip install tensorflow torchinfo joblib kagglehub
```

TensorFlow is only required for the spiral notebook and the activation-function notebook. `torchinfo` is only required for the `nn.Module` demos. `kagglehub` is only required for the two Week 06 conceptual sessions. There is no lockfile, the notebooks do not pin versions, and I have not re-run this install outside the environment that produced the saved outputs. After cloning, replace every `/content/...` path with the real file in the repo before running a notebook.

### Datasets and assets

| Dataset / asset | Used in | Included or downloaded | Access / path notes |
|---|---|---|---|
| Iris | Module 01, Optional mini project | `sklearn.datasets.load_iris` | No file in the repo |
| `make_regression` | Module 04 | Generated in the notebook | 100 samples, 1 feature, `random_state=13` |
| `make_blobs` | Assignment 01 | Generated in the notebook | 200 samples, 2 centers, `random_state=42`. Not real satellite images |
| Breast Cancer Wisconsin CSV | Modules 11, 12, and 15 | Downloaded when the notebook runs | `raw.githubusercontent.com/gscdit/Breast-Cancer-Detection/.../data.csv` |
| Digit `.npy` arrays | Week 04 conceptual session | **Included** next to the notebooks | 60,000 and 10,000 images, shape `(N, 28, 28)`, labels 0–9, values 0–255. Notebooks still point at `/content/` |
| Fashion-MNIST | Midterm | `torchvision` download | 60,000 / 10,000 official split. Not the same file as `fmnist_small.csv` |
| `fmnist_small.csv` | Module 16 and Module 19 | **Included** in Module 16 (6,000 × 785) | Notebooks read `/content/fmnist_small.csv`. The course GPU notebook’s saved scores do not match this row count |
| `make_moons` | Module 17 | Generated in the notebook | 250 samples, `noise=0.05`, `random_state=42` |
| Sign Language MNIST CSVs | Week 05 conceptual session | **Not included** | Notebook expects `sign_mnist_train.csv` and `sign_mnist_test.csv` in the working directory |
| `Social_Network_Ads.csv` | Module 18, and the Week 06 conceptual sessions | **Included** in Module 18 (400 rows). Conceptual sessions also download it with `kagglehub` | Module 18 reads `/content/Social_Network_Ads.csv`. Only Age and EstimatedSalary are used as features |
| KMNIST | Final exam | `torchvision` download | 28×28, 10 character classes |
| *Grokking Deep Learning* PDF | Module 01 | Included as course reading | Reference book, not a notebook input |
| Ioffe & Szegedy batch-norm paper | Module 19 | Included as `research_paper.pdf` | Reading, not loaded by code |

No trained `.pth` checkpoints are stored. Anything a notebook prints as accuracy is an output frozen in the `.ipynb`, not a model you can reload.

---

## Acknowledgements

- **[Phitron](https://phitron.io)** for the course structure, the `Phitron_` notebooks, the notes, the practice sheets, and the exam papers.
- **Andrew W. Trask, *Grokking Deep Learning***, included as a PDF in Week 01 Module 01.
- **Sergey Ioffe and Christian Szegedy**, *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift* (arXiv:1502.03167), included in Module 19.
- **Dataset sources named in the notebooks:** Iris and the `make_*` generators via scikit-learn; the Breast Cancer table via the `gscdit/Breast-Cancer-Detection` GitHub file the notebooks download; Fashion-MNIST and KMNIST via `torchvision`; Social Network Ads via `kagglehub` dataset `rakeshrau/social-network-ads` (and the CSV copy in Module 18). `Usefull Links .docx` also points at a Kaggle mirror of that ads table.
- **Short reading lists** stored as `.docx` files: GeeksforGeeks and a Desmos softmax graph (Module 05), GeeksforGeeks on batch versus stochastic and mini-batch gradient descent (Module 13), the PyTorch data tutorial and related posts (Module 15), and a Medium note on covariate shift (Module 19).
