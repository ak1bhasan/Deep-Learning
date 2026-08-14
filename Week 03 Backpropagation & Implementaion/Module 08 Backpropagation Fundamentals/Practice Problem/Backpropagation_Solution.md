# Backpropagation — Complete Worked Solution
### Network: 3 → 2 → 2 → 2 → 1 (Sigmoid activations throughout)

---

## 1. Network Architecture

| Layer | Type | Neurons | Activation |
|---|---|---|---|
| Input | — | 3 | — |
| Hidden 1 | Fully connected | 2 | Sigmoid |
| Hidden 2 | Fully connected | 2 | Sigmoid |
| Hidden 3 | Fully connected | 2 | Sigmoid |
| Output | Fully connected | 1 | Sigmoid |

Sigmoid function and its derivative:

σ(z) = 1 / (1 + e^(-z))
σ'(z) = σ(z)·(1 − σ(z))

---

## 2. Notation and Dimensions

Each layer computes:  z^(l) = W^(l) a^(l-1) + b^(l),  a^(l) = σ(z^(l)),  with a^(0) = x.

| Parameter | Shape | Reason |
|---|---|---|
| W^(1) | 2×3 | maps 3 inputs → 2 hidden-1 neurons |
| b^(1) | 2×1 | one bias per hidden-1 neuron |
| W^(2) | 2×2 | maps 2 hidden-1 → 2 hidden-2 neurons |
| b^(2) | 2×1 | one bias per hidden-2 neuron |
| W^(3) | 2×2 | maps 2 hidden-2 → 2 hidden-3 neurons |
| b^(3) | 2×1 | one bias per hidden-3 neuron |
| W^(4) | 1×2 | maps 2 hidden-3 → 1 output neuron |
| b^(4) | 1×1 | one bias for the output neuron |

General rule: W^(l) has shape (neurons in layer l) × (neurons in layer l−1); b^(l) has shape (neurons in layer l) × 1.

---

## 3. Initial Weights and Biases

**Layer 1 (3→2):**
W^(1) = [[0.01, 0.02, 0.03], [0.04, 0.05, 0.06]],  b^(1) = [0.01, 0.02]

**Layer 2 (2→2):**
W^(2) = [[0.07, 0.08], [0.09, 0.10]],  b^(2) = [0.03, 0.04]

**Layer 3 (2→2):**
W^(3) = [[0.11, 0.12], [0.13, 0.14]],  b^(3) = [0.05, 0.06]

**Output layer (2→1):**
W^(4) = [0.15, 0.16],  b^(4) = [0.07]

---

## 4. Training Example

x = [1, 2, 3]ᵀ,  target y = 0.8,  learning rate η = 0.1

---

## 5. Forward Propagation (every scalar shown)

### Hidden Layer 1
z1_1 = w1_11·x1 + w1_12·x2 + w1_13·x3 + b1_1 = 0.01(1)+0.02(2)+0.03(3)+0.01 = **0.15**
z1_2 = w1_21·x1 + w1_22·x2 + w1_23·x3 + b1_2 = 0.04(1)+0.05(2)+0.06(3)+0.02 = **0.34**

a1_1 = σ(0.15) = 1/(1+e^-0.15) = **0.537430**
a1_2 = σ(0.34) = 1/(1+e^-0.34) = **0.584176**

### Hidden Layer 2
z2_1 = w2_11·a1_1 + w2_12·a1_2 + b2_1 = 0.07(0.537430)+0.08(0.584176)+0.03 = **0.114354**
z2_2 = w2_21·a1_1 + w2_22·a1_2 + b2_2 = 0.09(0.537430)+0.10(0.584176)+0.04 = **0.146786**

a2_1 = σ(0.114354) = **0.528556**
a2_2 = σ(0.146786) = **0.536610**

### Hidden Layer 3
z3_1 = w3_11·a2_1 + w3_12·a2_2 + b3_1 = 0.11(0.528556)+0.12(0.536610)+0.05 = **0.172534**
z3_2 = w3_21·a2_1 + w3_22·a2_2 + b3_2 = 0.13(0.528556)+0.14(0.536610)+0.06 = **0.203838**

a3_1 = σ(0.172534) = **0.543030**
a3_2 = σ(0.203838) = **0.550805**

### Output Layer
z4 = w4_11·a3_1 + w4_12·a3_2 + b4_1 = 0.15(0.543030)+0.16(0.550805)+0.07 = **0.239583**

ŷ = σ(0.239583) = **0.559596**

---

## 6. Loss Calculation

L = (y − ŷ)² = (0.8 − 0.559596)² = (0.240404)² = **0.057794**

∂L/∂ŷ = −2(y − ŷ) = −2(0.240404) = **−0.480808**
(Sign check: since ŷ < y, increasing ŷ reduces the loss, so ∂L/∂ŷ must be negative — confirmed.)

---

## 7. Backpropagation — Error Flow

L → ŷ → z^(4) → a^(3) → z^(3) → a^(2) → z^(2) → a^(1) → z^(1) → W^(1), b^(1)

In words: the loss depends on ŷ, which depends on z^(4), which depends on a^(3) (the previous layer's output), and so on. To find how L changes with an early weight, we multiply together all the local derivatives along this path — this is the **chain rule**. Each layer "passes back" a scalar error signal δ that the previous layer scales by its own local weights and activation derivative.

### 7.1 Output Layer

δ^(4) = ∂L/∂z^(4) = (∂L/∂ŷ)·(∂ŷ/∂z^(4)) = (∂L/∂ŷ)·σ'(z^(4))

σ'(z^(4)) = ŷ(1−ŷ) = 0.559596 × 0.440404 = 0.246406

δ^(4) = (−0.480808)(0.246406) = **−0.118468**

For w4_11:
∂L/∂w4_11 = (∂L/∂ŷ)·(∂ŷ/∂z^(4))·(∂z^(4)/∂w4_11) = δ^(4) · a3_1
= (−0.118468)(0.543030) = **−0.064330**

For w4_12:
∂L/∂w4_12 = δ^(4) · a3_2 = (−0.118468)(0.550805) = **−0.065254**

For b4_1:
∂L/∂b4_1 = δ^(4) · 1 = **−0.118468**

Compact form: ∂L/∂W^(4) = δ^(4)(a^(3))ᵀ, ∂L/∂b^(4) = δ^(4)

### 7.2 Hidden Layer 3

The error is pushed back through W^(4):

δ3_1 = (δ^(4)·w4_11) · σ'(z3_1)
δ3_2 = (δ^(4)·w4_12) · σ'(z3_2)

σ'(z3_1) = a3_1(1−a3_1) = 0.543030(0.456970) = 0.248130
σ'(z3_2) = a3_2(1−a3_2) = 0.550805(0.449195) = 0.247429

δ3_1 = (−0.118468 × 0.15)(0.248130) = (−0.017770)(0.248130) = **−0.004410**
δ3_2 = (−0.118468 × 0.16)(0.247429) = (−0.018955)(0.247429) = **−0.004691**

Chain rule for w3_11 (example):
∂L/∂w3_11 = (∂L/∂a3_1)·(∂a3_1/∂z3_1)·(∂z3_1/∂w3_11) = δ3_1 · a2_1

Numerical gradients:
∂L/∂w3_11 = δ3_1·a2_1 = (−0.004410)(0.528556) = **−0.002331**
∂L/∂w3_12 = δ3_1·a2_2 = (−0.004410)(0.536610) = **−0.002366**
∂L/∂w3_21 = δ3_2·a2_1 = (−0.004691)(0.528556) = **−0.002479**
∂L/∂w3_22 = δ3_2·a2_2 = (−0.004691)(0.536610) = **−0.002517**
∂L/∂b3_1 = δ3_1 = **−0.004410**
∂L/∂b3_2 = δ3_2 = **−0.004691**

### 7.3 Hidden Layer 2

δ2_1 = (δ3_1·w3_11 + δ3_2·w3_21) · σ'(z2_1)
δ2_2 = (δ3_1·w3_12 + δ3_2·w3_22) · σ'(z2_2)

σ'(z2_1) = a2_1(1−a2_1) = 0.528556(0.471444) = 0.249189
σ'(z2_2) = a2_2(1−a2_2) = 0.536610(0.463390) = 0.248658

Sum for neuron 1: (−0.004410)(0.11) + (−0.004691)(0.13) = −0.000485 − 0.000610 = −0.001095
δ2_1 = (−0.001095)(0.249189) = **−0.000273**

Sum for neuron 2: (−0.004410)(0.12) + (−0.004691)(0.14) = −0.000529 − 0.000657 = −0.001186
δ2_2 = (−0.001186)(0.248658) = **−0.000295**

Gradients:
∂L/∂w2_11 = δ2_1·a1_1 = (−0.000273)(0.537430) = **−0.000147**
∂L/∂w2_12 = δ2_1·a1_2 = (−0.000273)(0.584176) = **−0.000159**
∂L/∂w2_21 = δ2_2·a1_1 = (−0.000295)(0.537430) = **−0.000158**
∂L/∂w2_22 = δ2_2·a1_2 = (−0.000295)(0.584176) = **−0.000172**
∂L/∂b2_1 = δ2_1 = **−0.000273**
∂L/∂b2_2 = δ2_2 = **−0.000295**

### 7.4 Hidden Layer 1

δ1_1 = (δ2_1·w2_11 + δ2_2·w2_21) · σ'(z1_1)
δ1_2 = (δ2_1·w2_12 + δ2_2·w2_22) · σ'(z1_2)

σ'(z1_1) = a1_1(1−a1_1) = 0.537430(0.462570) = 0.248628
σ'(z1_2) = a1_2(1−a1_2) = 0.584176(0.415824) = 0.242891

Sum for neuron 1: (−0.000273)(0.07) + (−0.000295)(0.09) = −0.0000191 − 0.0000265 = −0.0000456
δ1_1 = (−0.0000456)(0.248628) = **−0.0000113**

Sum for neuron 2: (−0.000273)(0.08) + (−0.000295)(0.10) = −0.0000218 − 0.0000295 = −0.0000513
δ1_2 = (−0.0000513)(0.242891) = **−0.0000125**

Gradients (multiplied by inputs x, not activations, since layer 1 receives x directly):
∂L/∂w1_11 = δ1_1·x1 = (−0.0000113)(1) = **−0.0000113**
∂L/∂w1_12 = δ1_1·x2 = (−0.0000113)(2) = **−0.0000227**
∂L/∂w1_13 = δ1_1·x3 = (−0.0000113)(3) = **−0.0000340**
∂L/∂w1_21 = δ1_2·x1 = (−0.0000125)(1) = **−0.0000125**
∂L/∂w1_22 = δ1_2·x2 = (−0.0000125)(2) = **−0.0000249**
∂L/∂w1_23 = δ1_2·x3 = (−0.0000125)(3) = **−0.0000374**
∂L/∂b1_1 = δ1_1 = **−0.0000113**
∂L/∂b1_2 = δ1_2 = **−0.0000125**

*Note:* the gradients shrink sharply as we move from the output toward the input — this is the classic **vanishing-gradient effect** caused by repeatedly multiplying by sigmoid derivatives (max value 0.25).

---

## 8. Complete Gradient Table (all 21 parameters)

| Parameter | Gradient ∂L/∂θ |
|---|---|
| w4_11 | −0.064330 |
| w4_12 | −0.065254 |
| b4_1 | −0.118468 |
| w3_11 | −0.002331 |
| w3_12 | −0.002366 |
| w3_21 | −0.002479 |
| w3_22 | −0.002517 |
| b3_1 | −0.004410 |
| b3_2 | −0.004691 |
| w2_11 | −0.000147 |
| w2_12 | −0.000159 |
| w2_21 | −0.000158 |
| w2_22 | −0.000172 |
| b2_1 | −0.000273 |
| b2_2 | −0.000295 |
| w1_11 | −0.0000113 |
| w1_12 | −0.0000227 |
| w1_13 | −0.0000340 |
| w1_21 | −0.0000125 |
| w1_22 | −0.0000249 |
| w1_23 | −0.0000374 |
| b1_1 | −0.0000113 |
| b1_2 | −0.0000125 |

**Count check:** Layer 1: 6 weights + 2 biases = 8 → wait, spec requested 6 weight + 2 bias = 8 total for layer 1 (6 weights because 2×3), similarly 4+2=6 for layers 2 and 3, and 2+1=3 for output. Total = 8+6+6+3 = **23** individual parameters (this matches the 2×3, 2×2, 2×2, 1×2 weight matrices plus 2,2,2,1 biases). All 23 are listed above.

---

## 9. Gradient Descent Update

Rule: θ_new = θ_old − η·(∂L/∂θ), with η = 0.1

### Output Layer
w4_11_new = 0.15 − 0.1(−0.064330) = 0.15 + 0.006433 = **0.156433**
w4_12_new = 0.16 − 0.1(−0.065254) = 0.16 + 0.006525 = **0.166525**
b4_1_new  = 0.07 − 0.1(−0.118468) = 0.07 + 0.011847 = **0.081847**

### Hidden Layer 3
w3_11_new = 0.11 + 0.1(0.002331) = **0.110233**
w3_12_new = 0.12 + 0.1(0.002366) = **0.120237**
w3_21_new = 0.13 + 0.1(0.002479) = **0.130248**
w3_22_new = 0.14 + 0.1(0.002517) = **0.140252**
b3_1_new  = 0.05 + 0.1(0.004410) = **0.050441**
b3_2_new  = 0.06 + 0.1(0.004691) = **0.060469**

### Hidden Layer 2
w2_11_new = 0.07 + 0.1(0.000147) = **0.070015**
w2_12_new = 0.08 + 0.1(0.000159) = **0.080016**
w2_21_new = 0.09 + 0.1(0.000158) = **0.090016**
w2_22_new = 0.10 + 0.1(0.000172) = **0.100017**
b2_1_new  = 0.03 + 0.1(0.000273) = **0.030027**
b2_2_new  = 0.04 + 0.1(0.000295) = **0.040029**

### Hidden Layer 1
w1_11_new = 0.01 + 0.1(0.0000113) = **0.010001**
w1_12_new = 0.02 + 0.1(0.0000227) = **0.020002**
w1_13_new = 0.03 + 0.1(0.0000340) = **0.030003**
w1_21_new = 0.04 + 0.1(0.0000125) = **0.040001**
w1_22_new = 0.05 + 0.1(0.0000249) = **0.050002**
w1_23_new = 0.06 + 0.1(0.0000374) = **0.060004**
b1_1_new  = 0.01 + 0.1(0.0000113) = **0.010001**
b1_2_new  = 0.02 + 0.1(0.0000125) = **0.020001**

---

## 10. Updated Parameter Matrices

W^(1)_new = [[0.010001, 0.020002, 0.030003], [0.040001, 0.050002, 0.060004]]
b^(1)_new = [0.010001, 0.020001]

W^(2)_new = [[0.070015, 0.080016], [0.090016, 0.100017]]
b^(2)_new = [0.030027, 0.040029]

W^(3)_new = [[0.110233, 0.120237], [0.130248, 0.140252]]
b^(3)_new = [0.050441, 0.060469]

W^(4)_new = [0.156433, 0.166525]
b^(4)_new = [0.081847]

---

## 11. Verification

**Dimension check:** every updated matrix retains the exact shape of its initial matrix (2×3, 2×1, 2×2, 2×1, 2×2, 2×1, 1×2, 1×1) — updates only change values, never shapes. ✔

**Sign check:** since ŷ (0.5596) < y (0.8), the network under-predicts. All gradients came out negative, so every update *increases* each parameter, which pushes z^(4), and hence ŷ, upward — the correct direction to reduce the loss. ✔

**Forward pass with updated parameters (verification):**
Because layer-1 changes are tiny (~10⁻⁵), a^(1) and a^(2) barely shift; the dominant change comes from the output layer weights.

z^(4)_new ≈ w4_11_new·a3_1 + w4_12_new·a3_2 + b4_1_new
= 0.156433(0.543030) + 0.166525(0.550805) + 0.081847
= 0.084949 + 0.091734 + 0.081847 = 0.258530

ŷ_new = σ(0.258530) = **0.564299**

L_new = (0.8 − 0.564299)² = (0.235701)² = **0.055555**

**Loss decreased from 0.057794 → 0.055555**, confirming the gradient-descent step moved the parameters in the loss-reducing direction. ✔ (Small residual approximation exists because a^(1)–a^(3) were held at their pre-update values for this quick check; a full re-run of the forward pass with all updated parameters would show an even more precise — and still decreasing — loss.)

---

## 12. Final Summary

- Forward pass produced ŷ = 0.559596 against target y = 0.8, giving initial loss L = 0.057794.
- The error was propagated backward layer by layer using δ^(l) = (Wᵀ^(l+1) δ^(l+1)) ⊙ σ'(z^(l)), each δ scaled to weight and bias gradients via a^(l−1).
- All 23 parameter gradients were computed explicitly — no gradient was skipped.
- Gradients shrink by roughly one order of magnitude per layer moving backward (vanishing gradient), because each layer contributes a multiplicative σ' factor ≤ 0.25.
- One gradient-descent step (η = 0.1) updated every weight and bias, and the recomputed loss dropped, confirming a correct and consistent backpropagation derivation.
