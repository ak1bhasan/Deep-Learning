# Perceptron: Labels, Update Rule, and Gradient Descent — Full Derivation

### Q1. Why does the perceptron algorithm use class labels −1 and +1 instead of 0 and 1? Explain how this choice affects the perceptron update rule.

This question is really about comparing two choices of `Activation Function` for the perceptron, both built on the same weighted sum $`z \equiv wx + b`$.

```math
a_{0,1}(z) \equiv
\begin{cases}
1, & \text{if } z \geq 0 \\
0, & \text{if } z < 0
\end{cases}
\qquad\qquad
a_{\pm 1}(z) \equiv
\begin{cases}
1, & \text{if } z \geq 0 \\
-1, & \text{if } z < 0
\end{cases}
```

Each choice forces a different `Loss Function`, and consequently a different `Learning Rule`, once we apply Gradient Descent to it.

**Case 1 — using $`a_{\pm 1}(z)`$**

The standard loss here is the `hinge loss`:

```math
L_{W,b}(x) \equiv \max(0, -yz)
```

- Correct classification: $`y`$ and $`z`$ agree in sign $`\implies yz \geq 0 \implies -yz \leq 0 \implies L \equiv 0`$.
- Misclassification: $`y`$ and $`z`$ disagree in sign $`\implies yz < 0 \implies -yz > 0 \implies L \equiv -yz \equiv -y(wx+b)`$.

Applying Gradient Descent, $`w_{new} = w_{old} - \eta \frac{\partial L}{\partial w}`$:

```math
w_{new} =
\begin{cases}
w_{old} - \eta \frac{\partial}{\partial w}(0), & \text{Correct Classification} \\[4pt]
w_{old} - \eta \frac{\partial}{\partial w}\big(-y(wx+b)\big), & \text{Misclassification}
\end{cases}
=
\begin{cases}
w_{old}, & \text{Correct Classification} \\
w_{old} + \eta y x, & \text{Misclassification}
\end{cases}
```

```math
b_{new} =
\begin{cases}
b_{old}, & \text{Correct Classification} \\
b_{old} + \eta y, & \text{Misclassification}
\end{cases}
```

**Case 2 — using $`a_{0,1}(z)`$**

A standard loss here is:

```math
L_{W,b}(x) \equiv (y - \hat{y})\,z
```

- Correct classification: $`(y-\hat{y}) = 0 \implies L \equiv 0`$. Either Gradient Ascent or Descent is fine here since the loss is flat regardless.
- Misclassification: $`(y-\hat{y}) \neq 0`$, so $`L \equiv (y-\hat{y})(wx+b)`$.
  - False Positive ($`\hat y=1, y=0`$): $`y - \hat y = -1 \implies L < 0`$ (assuming $`z>0`$ since the model predicted class 1). This case needs `Gradient Ascent` to move $`L`$ toward $`0`$.
  - False Negative ($`\hat y=0, y=1`$): $`y - \hat y = 1 \implies L < 0`$ (since $`z<0`$ here). This also needs `Gradient Ascent`.

Applying Gradient **Ascent**, $`w_{new} = w_{old} + \eta \frac{\partial L}{\partial w}`$:

```math
w_{new} =
\begin{cases}
w_{old}, & \text{Correct Classification} \\
w_{old} + \eta (y-\hat y) x, & \text{Misclassification}
\end{cases}
\qquad
b_{new} =
\begin{cases}
b_{old}, & \text{Correct Classification} \\
b_{old} + \eta (y-\hat y), & \text{Misclassification}
\end{cases}
```

**Comparison**

| | $`a_{0,1}(z)`$ | $`a_{\pm 1}(z)`$ |
|---|---|---|
| Loss Function | $`L_{W,b}(x) \equiv (y-\hat y)z`$ | $`L_{W,b}(x) \equiv \max(0, -yz)`$ |
| Learning Rule (w) | $`w_{new} = w_{old} + \eta(y-\hat y)x`$ | $`w_{new} = w_{old} + \eta y x`$ |
| Learning Rule (b) | $`b_{new} = b_{old} + \eta(y-\hat y)`$ | $`b_{new} = b_{old} + \eta y`$ |
| Optimization direction | Gradient Ascent | Gradient Descent |

So the choice of class labels doesn't just relabel the classes — it changes the shape of the loss function itself, which in turn decides whether the "natural" optimization is ascent or descent, and whether the update needs an explicit error term $`(y-\hat y)`$ or gets it for free from the sign of $`y`$.

---

### Q2. Gradient descent usually updates weights by subtracting the gradient. Why does the perceptron update rule add the term ηyx instead?

It does still subtract the gradient — that part never changes. What changes is the sign already baked into the loss function itself, in the $`a_{\pm 1}(z)`$ case.

With $`a_{\pm 1}(z)`$ as the activation, the standard loss is:

```math
L_{W,b}(x) \equiv \max(0,-yz)
```

Its gradients, in the region where the example is misclassified (where the loss is active and non-zero), are:

```math
\frac{\partial L}{\partial w} = -yx \qquad\qquad \frac{\partial L}{\partial b} = -y
```

Plugging these into ordinary Gradient Descent:

```math
w_{new} = w_{old} - \eta\frac{\partial L}{\partial w} = w_{old} - \eta(-yx) = w_{old} + \eta y x
```

The "+" in $`\eta y x`$ isn't a departure from Gradient Descent — it's exactly what Gradient Descent produces once the negative sign inside the gradient (which itself comes from the $`-yz`$ inside the hinge loss) cancels against the negative sign that Gradient Descent always applies. Two negatives, one positive result.

---

### Q3 (optional). Explain in your own words how the perceptron learning rule can be seen as a form of gradient descent. Also mention one limitation of the perceptron loss function.

**As a form of gradient descent:**
The Perceptron Learning Rule isn't a separate, unrelated algorithm that merely resembles Gradient Descent — it's a special case of Gradient Descent, derived by applying the general Gradient Descent optimization algorithm directly to the perceptron's own loss function, $`L_{W,b}(x) \equiv \max(0,-yz)`$, exactly as shown in the Q1/Q2 derivation. Because that loss is exactly $`0`$ for every correctly classified point, its gradient there is also $`0`$, so the general update formula $`w_{new} = w_{old} - \eta \frac{\partial L}{\partial w}`$ naturally collapses to "no change" for correct predictions and to "$`w_{old} + \eta y x`$" for misclassified ones — without needing any special-cased rule bolted on separately. The perceptron rule is what falls out of ordinary Gradient Descent once you commit to this particular hinge-style loss.

**One limitation of the perceptron loss function:**
$`L_{W,b}(x) \equiv \max(0,-yz)`$ becomes exactly $`0`$ the moment a point crosses to the correct side of the boundary — however marginally. It has no further gradient pushing the boundary away from that point to create a safety margin. This means training can stop adjusting the boundary while it's still sitting right against some training points, with no notion of "how confidently correct" the classification is. A hinge loss with margin (as used in Support Vector Machines, $`\max(0, 1-yz)`$) fixes exactly this, by only reaching zero once a point is correctly classified *and* comfortably past a margin, not just barely across the line.
