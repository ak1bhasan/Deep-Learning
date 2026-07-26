# Practice Problem Solutions

## Bias

### **Q: What is a bias in a neural network?**
**Solution:** A bias is a learnable constant added to the weighted sum of inputs. It shifts the activation function so the model is not forced to pass through the origin.

### **Q: Why do we add bias to a perceptron?**
**Solution:** Adding bias lets the decision boundary shift freely in feature space, which allows the perceptron to fit real-world data patterns more accurately.

---

## Weights

### **Q: What are weights in a perceptron?**
**Solution:** Weights are numerical parameters that control the importance (strength) of each input feature in the weighted sum.

### **Q: How do weights affect the output of a perceptron?**
**Solution:** Each weight scales its corresponding input. Larger absolute weights push that feature’s contribution harder toward (or against) activation, which shapes the final decision.

---

## Perceptron Output

### **Q: A perceptron has inputs $x_1 = 2$, $x_2 = 3$, weights $w_1 = 0.5$, $w_2 = 1$, and bias $b = 1$. Calculate the weighted sum (before activation).**
**Solution:**

Given:

* $x_1 = 2$
* $x_2 = 3$
* $w_1 = 0.5$
* $w_2 = 1$
* $b = 1$

$$
\begin{aligned}
z &= x_1 w_1 + x_2 w_2 + b \\
  &= (2)(0.5) + (3)(1) + 1 \\
  &= 1 + 3 + 1 \\
  &= 5
\end{aligned}
$$

The weighted sum is $z = 5$.

---

## Step Function

### **Q: What is the step function and how does it work?**
**Solution:** The step function (binary step) maps a continuous input to one of two fixed outputs—typically $0$ or $1$—using a threshold.

**How it works:** Compare the input $z$ to a threshold (usually $0$):

* If $z < 0$, the output is $0$.
* If $z \geq 0$, the output is $1$.

**Mathematical formula:**

$$
\text{Output} =
\begin{cases}
1 & \text{if } z \geq 0 \\
0 & \text{if } z < 0
\end{cases}
$$

---

## Activation Function

### **Q: Why do we need an activation function in a perceptron?**
**Solution:** An activation function turns the unbounded weighted sum into a decision signal—whether the neuron should fire—and maps that sum into a bounded output range (for example, $\{0, 1\}$ for the step function).

### **Q: Name three common activation functions.**
**Solution:** ReLU, Sigmoid, and Tanh.

---

## Perceptron Architecture Diagram

The diagram below shows a single-layer perceptron: inputs and weights combine with an explicit bias at a summation node, then pass through an activation function.

![Perceptron Model Architecture Diagram](perceptron.webp)
