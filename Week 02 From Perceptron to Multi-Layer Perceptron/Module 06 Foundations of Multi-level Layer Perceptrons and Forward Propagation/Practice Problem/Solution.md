# Module 06 | Foundations of Multi-Layer Perceptrons and Forward Propagation

## My Solution to Practice Problems

**Table of Contents**
- [Q01. What happens if we stack multiple linear layers without activation?](#q01-what-happens-if-we-stack-multiple-linear-layers-without-activation)
- [Q02. What kind of decision boundary does an MLP create?](#q02-what-kind-of-decision-boundary-does-an-mlp-create)
- [Q03. What are the main components of a neural network?](#q03-what-are-the-main-components-of-a-neural-network)
- [Q04. What is the difference between input layer, hidden layer, and output layer?](#q04-what-is-the-difference-between-input-layer-hidden-layer-and-output-layer)
- [Q05. What is forward propagation?](#q05-what-is-forward-propagation)

---

### Q01. What happens if we stack multiple linear layers without activation?

Without a non-linear activation function, a layer just performs an affine transformation: multiply the input by a weight matrix and add a bias. If we stack several such layers back to back, each layer's output simply feeds into the next matrix multiplication — and a chain of matrix multiplications is mathematically still just one matrix multiplication.

Concretely, say layer 1 produces $O^{(1)} = W^{(1)}X + B^{(1)}$, and we feed that into layer 2:

$$O^{(2)} = W^{(2)}O^{(1)} + B^{(2)} = W^{(2)}(W^{(1)}X + B^{(1)}) + B^{(2)} = (W^{(2)}W^{(1)})X + (W^{(2)}B^{(1)} + B^{(2)})$$

Notice this has the exact same shape as a single linear layer — a combined weight matrix times $X$, plus a combined bias. Extend this to $n$ stacked linear layers, and the whole thing still reduces to one effective weight matrix and one effective bias.

So the takeaway is simple: **stacking linear layers without activation gains us nothing**. No matter how deep the stack, it's collapsible into a single linear layer, and the network's expressive power stays exactly the same as one layer would give us. This is precisely why non-linear activations are essential — they're what stop the stack from collapsing and actually let depth contribute something.

---

### Q02. What kind of decision boundary does an MLP create?

An MLP (Multi-Layer Perceptron) can create **non-linear decision boundaries**, and how complex those boundaries can get depends on its architecture:

- A single-layer perceptron with no hidden layer can only draw a straight line, or a flat hyperplane in higher dimensions — a strictly linear boundary.
- Once we add hidden layers with non-linear activations (ReLU, tanh, sigmoid, etc.), an MLP becomes capable of learning curved, irregular, even disconnected boundaries.
- The more hidden units and hidden layers an MLP has, the more intricate a boundary it can represent — assuming there's enough data and proper training to support that capacity.

**In short:** an MLP gives us flexible, non-linear decision boundaries, not just straight lines.

---

### Q03. What are the main components of a neural network?

1. **Input Layer**
   - Receives the input features.
   - Typically has one neuron per input feature.

2. **Hidden Layer(s)**
   - Where the actual computation and pattern extraction happens.
   - A network can have one hidden layer or many.

3. **Neurons (Nodes)**
   - The basic computing units of the network.
   - Each neuron computes a weighted sum of its inputs plus a bias, then applies an activation function to that sum.

4. **Weights**
   - Control how strongly each connection between neurons influences the next neuron.
   - These are the main parameters the network learns during training.

5. **Biases**
   - Shift a neuron's activation threshold up or down.
   - Give the model extra flexibility that weights alone can't provide.

6. **Activation Functions**
   - Introduce non-linearity into the network.
   - Common choices include:
     - ReLU
     - Sigmoid
     - Tanh
     - Softmax (typically used in the output layer for multi-class classification)

7. **Output Layer**
   - Produces the network's final prediction.
   - Its structure depends on the task:
     - Regression → one linear neuron
     - Binary classification → one sigmoid neuron
     - Multi-class classification → multiple softmax neurons

8. **Loss Function**
   - Quantifies how far off the predictions are from the true values.
   - Common examples:
     - Mean Squared Error (MSE)
     - Binary Cross-Entropy
     - Categorical Cross-Entropy

9. **Optimizer**
   - Adjusts weights and biases to bring the loss down.
   - Common examples:
     - Gradient Descent
     - SGD
     - Adam
     - RMSprop

> **In one sentence:** A neural network is made of layers of neurons linked by weights and biases, using activation functions to capture non-linear patterns, and trained by minimizing a loss function through an optimizer.

---

### Q04. What is the difference between input layer, hidden layer, and output layer?

**Input Layer**
- Takes in the raw data as-is.
- Doesn't perform any real computation on it.
- Simply forwards the input values on to the first hidden layer.
- Number of neurons equals the number of input features.

*Example: if each image is represented as 784 pixel values, the input layer has 784 neurons.*

**Hidden Layer**
- Actually processes and transforms the input.
- Learns intermediate features and patterns from the data.
- Each neuron here computes a weighted sum, adds a bias, and applies an activation function.
- A network may have one hidden layer or several stacked together.

*Example: a hidden layer might learn to detect edges, shapes, textures, or more abstract concepts from an image.*

**Output Layer**
- Delivers the network's final prediction.
- Its structure is shaped by the task at hand.

Examples:
- **Regression:** 1 linear neuron
- **Binary Classification:** 1 sigmoid neuron
- **Multi-class Classification:** one softmax neuron per class

**Summary**

| Layer | Purpose | Learns? |
|---|---|---|
| Input Layer | Receives input features | No |
| Hidden Layer | Extracts intermediate representations | Yes |
| Output Layer | Produces the final prediction | Yes |

---

### Q05. What is forward propagation?

Forward propagation is how input data travels through a neural network — starting at the input layer, passing through each hidden layer in turn, and finally reaching the output layer where a prediction is produced.

At every neuron along the way, the same three steps repeat:

1. Compute the weighted sum of the inputs coming in, plus a bias term.
2. Apply an activation function to that sum.
3. Pass the resulting value forward as input to the next layer.

This chain of computation keeps going, layer after layer, until it reaches the output layer and produces the network's prediction.

```
Input
  |
Input Layer
  |
Hidden Layer 1
  |
Hidden Layer 2
  |
 ...
  |
Output Layer
  |
Prediction
```

Forward propagation serves a couple of key purposes:

- It generates the network's actual prediction for a given input.
- That prediction gets compared against the true target to compute the loss.
- The values computed during this forward pass are also what backpropagation needs later on, to work out the gradients used by the optimizer to update the network's parameters.

> **In short:** forward propagation is simply the process of computing a neural network's output from its input, using the network's current weights and biases.
