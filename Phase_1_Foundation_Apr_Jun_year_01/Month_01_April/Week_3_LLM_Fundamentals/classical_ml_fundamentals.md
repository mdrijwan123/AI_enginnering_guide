# Classical ML Fundamentals — The Complete Interview Guide
### Prerequisite Knowledge for Every AI Engineer Interview

> **Why this matters:** Even for LLM/AI Engineer roles, top-tier interviews include an ML Fundamentals round. You WILL be asked about bias-variance, gradient descent, regularisation, evaluation metrics, and classical algorithms. Candidates who ace the LLM round but stumble on "explain the bias-variance tradeoff" get rejected.

> **How to use this file:** Read top-to-bottom. Every concept builds on the previous one. The code is runnable Python — copy-paste and experiment. The interview Q&As at the end cover the exact questions asked.

---

## TABLE OF CONTENTS

1. [Supervised vs Unsupervised vs Reinforcement Learning](#part-1)
2. [Linear Regression from Scratch](#part-2)
3. [Gradient Descent — All Variants](#part-3)
4. [Logistic Regression](#part-4)
5. [The Bias-Variance Tradeoff](#part-5)
6. [Regularisation (L1, L2, ElasticNet, Dropout)](#part-6)
7. [Evaluation Metrics — The Complete Guide](#part-7)
8. [Cross-Validation](#part-8)
9. [Decision Trees](#part-9)
10. [Ensemble Methods (Random Forest, XGBoost, LightGBM)](#part-10)
11. [Support Vector Machines (SVM)](#part-11)
12. [K-Nearest Neighbours (KNN)](#part-12)
13. [Naive Bayes](#part-13)
14. [Clustering (K-Means, DBSCAN)](#part-14)
15. [Dimensionality Reduction (PCA, t-SNE)](#part-15)
16. [Feature Engineering](#part-16)
17. [Probability & Statistics for Interviews](#part-17)
18. [A/B Testing & Hypothesis Testing](#part-18)
19. [Interview Q&A — 50 Questions](#part-19)
20. [Further Resources](#part-20)

---

<a name="part-1"></a>
## Part 1 — The Three Types of Machine Learning

> 💡 **ELI5 (Explain Like I'm 5):** 
> 1. **Supervised:** You show a toddler 10 pictures of cats and say "kat". Then you show a dog and say "not kat". The toddler learns from being explicitly taught with labels.
> 2. **Unsupervised:** You give a toddler a box of mixed Legos. Without you saying a word, they sort them into red, blue, and yellow piles. They found patterns naturally.
> 3. **Reinforcement:** You give a toddler a video game controller. They press A and fall in lava (bad). They press B and get a coin (good). By trial and error, they learn to press B.

### 1.1 Supervised Learning

The model learns from **labelled** examples: (input, correct output) pairs.

```
Training data:
  Input (features)           Output (label)
  [3 bedrooms, 1500 sqft]   $300,000    (regression — continuous)
  [spam words, links]        SPAM        (classification — discrete)

The model learns: f(input) → output
At inference: given new input, predict output
```

**Two sub-types:**
| Type | Output | Loss Function | Examples |
|---|---|---|---|
| **Regression** | Continuous number | MSE, MAE | Price prediction, temperature |
| **Classification** | Discrete class | Cross-entropy, hinge | Spam detection, image class |

### 1.2 Unsupervised Learning

No labels. The model finds **patterns, structure, or groupings** in data.

```
Input: customer purchase data (no labels)
Output: "These customers form 5 natural groups" (clustering)
        "These 50 features can be compressed to 3 principal components" (dim reduction)
```

**Common tasks:**
- **Clustering:** K-Means, DBSCAN, hierarchical
- **Dimensionality reduction:** PCA, t-SNE, UMAP
- **Anomaly detection:** Isolation Forest, autoencoders
- **Association rules:** market basket analysis

### 1.3 Reinforcement Learning

An **agent** takes actions in an **environment** to maximise cumulative **reward**.

```
Agent (the model) → takes Action → Environment changes → returns Reward + new State
Agent learns: which actions lead to higher long-term reward

Examples:
  - AlphaGo: agent = model, action = place stone, reward = win/lose
  - RLHF: agent = LLM, action = generate token, reward = human preference score
  - Robotics: agent = robot, action = move arm, reward = grasp success
```

**Why this matters for LLM interviews:** RLHF (covered in Month 9) IS reinforcement learning. Understanding the basics of RL helps you explain PPO/DPO with confidence.

### 1.4 Semi-Supervised and Self-Supervised Learning

```
Semi-supervised: Small labelled set + large unlabelled set
  → Use model trained on labelled data to pseudo-label unlabelled data
  → Common in production where labelling is expensive

Self-supervised: Create labels FROM the data itself
  → LLM pre-training: predict next token (label = the actual next token)
  → BERT: mask a word, predict it (label = the masked word)
  → CLIP: pair (image, caption) — the pairing IS the supervision
  → This is what makes foundation models possible at scale
```

---

<a name="part-2"></a>
## Part 2 — Linear Regression: From Scratch

### 2.1 What It Does

> 💡 **ELI5 (Explain Like I'm 5):** Imagine mapping height to weight on a graph. The dots are scattered vaguely upwards. Linear Regression is the math magic to draw the "best possible straight line" through the exact middle of all those dots so you can predict the weight of someone based on their height.

Find the best straight line (or hyperplane) through data points.

```
y = w₁x₁ + w₂x₂ + ... + wₙxₙ + b

In vector form: y = w·x + b

where:
  w = weights (slope coefficients)
  b = bias (intercept)
  x = input features
  y = predicted output
```

**Visual intuition (1 feature):**
```
Price ($)
   |      ·  /
   |   ·  /·
   |  · /·
   | ·/ ·
   |/· 
   +--------→ Square footage
   
The line: price = w × sqft + b
Goal: find w, b that makes the line "best fit" the dots
```

### 2.2 The Loss Function: Mean Squared Error (MSE)

How do we define "best fit"? Minimise the average squared distance between predictions and actual values:

$$\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$

Where:
- $y_i$ = actual value (ground truth)
- $\hat{y}_i$ = predicted value ($w \cdot x_i + b$)
- $n$ = number of training examples

**Why squared?**
1. Makes all errors positive (a prediction that's $10 too high counts the same as $10 too low)
2. Penalises large errors more than small errors (squaring amplifies outlier errors)
3. Smooth and differentiable everywhere — gradient descent works well

**Other loss functions for regression:**
```
MAE (Mean Absolute Error) = (1/n) Σ|yᵢ - ŷᵢ|
  → More robust to outliers than MSE
  → Not differentiable at 0 (use Huber loss as compromise)

Huber Loss = MSE when error < δ, MAE when error ≥ δ
  → Best of both worlds

RMSE = √MSE
  → Same units as the target variable (easier to interpret)
```

### 2.3 Closed-Form Solution (Normal Equation)

For linear regression specifically, there's a direct formula:

$$w = (X^TX)^{-1}X^Ty$$

```python
import numpy as np

# Generate sample data
np.random.seed(42)
X = np.random.rand(100, 3)  # 100 samples, 3 features
y = X @ np.array([3, -2, 7]) + 1.5 + np.random.randn(100) * 0.5  # true weights + noise

# Add bias column (column of 1s)
X_b = np.c_[np.ones(len(X)), X]  # shape: (100, 4)

# Normal equation
w = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
print(f"Learned weights: {w}")
# Should be close to: [1.5, 3, -2, 7] (bias, w1, w2, w3)
```

**When to use normal equation vs gradient descent:**
| | Normal Equation | Gradient Descent |
|---|---|---|
| Complexity | O(n³) for matrix inverse | O(n × epochs) |
| Features > 10,000 | Too slow | Works fine |
| Training data > 100K | Still works but slow | Better |
| Non-linear models | Not applicable | Works for any differentiable model |

### 2.4 Linear Regression from Scratch (Gradient Descent)

```python
import numpy as np

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.lr = learning_rate
        self.n_iter = n_iterations
        self.weights = None
        self.bias = None
        self.losses = []
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Initialise parameters to zero
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        for i in range(self.n_iter):
            # Forward pass: predictions
            y_pred = X @ self.weights + self.bias
            
            # Compute loss (MSE)
            loss = np.mean((y - y_pred) ** 2)
            self.losses.append(loss)
            
            # Compute gradients (partial derivatives of MSE w.r.t. weights and bias)
            # d(MSE)/dw = -(2/n) * X^T(y - y_pred)
            # d(MSE)/db = -(2/n) * sum(y - y_pred)
            dw = -(2 / n_samples) * (X.T @ (y - y_pred))
            db = -(2 / n_samples) * np.sum(y - y_pred)
            
            # Update parameters (gradient descent step)
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
        
        return self
    
    def predict(self, X):
        return X @ self.weights + self.bias

# Usage
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=200, n_features=3, noise=10, random_state=42)

# Normalise features (critical for gradient descent to converge!)
X = (X - X.mean(axis=0)) / X.std(axis=0)

model = LinearRegression(learning_rate=0.01, n_iterations=1000)
model.fit(X, y)

print(f"Final MSE: {model.losses[-1]:.4f}")
print(f"Weights: {model.weights}")
```

**Key takeaway:** The gradient tells us which direction to move each weight to reduce the loss. We take small steps (learning rate) in that direction repeatedly.

---

<a name="part-3"></a>
## Part 3 — Gradient Descent: All Variants

### 3.1 The Core Algorithm

```
Repeat until convergence:
    1. Compute predictions: ŷ = f(x; θ)
    2. Compute loss: L(ŷ, y)
    3. Compute gradient: ∇L = ∂L/∂θ  (how each parameter affects loss)
    4. Update parameters: θ = θ - η × ∇L  (move opposite to gradient)
```

**Intuition:** You're blindfolded on a hilly landscape. The gradient is like feeling the slope under your feet. You step downhill. Repeat until you reach a valley (minimum).

### 3.2 Three Flavours

**Batch Gradient Descent:**
```python
# Uses ALL training data per step
for epoch in range(n_epochs):
    gradient = compute_gradient(X_train, y_train)  # ALL data points
    weights -= lr * gradient

# Pros: Stable, smooth convergence
# Cons: Slow for large datasets (must process all data per step)
# Memory: Must fit all data in memory
```

**Stochastic Gradient Descent (SGD):**
```python
# Uses ONE random training example per step
for epoch in range(n_epochs):
    shuffle(X_train, y_train)
    for x_i, y_i in zip(X_train, y_train):
        gradient = compute_gradient(x_i, y_i)  # Just 1 sample
        weights -= lr * gradient

# Pros: Fast per step, can escape local minima (noisy updates)
# Cons: Very noisy, may not converge smoothly
# Use when: Data is huge, online learning (streaming)
```

**Mini-Batch Gradient Descent (what everyone actually uses):**
```python
# Uses a BATCH of B training examples per step (typically B = 32, 64, 128, 256)
for epoch in range(n_epochs):
    for batch in create_batches(X_train, y_train, batch_size=64):
        gradient = compute_gradient(batch)
        weights -= lr * gradient

# Pros: Balanced between speed and stability
# Cons: Need to tune batch size
#
# THIS IS WHAT "SGD" MEANS IN PRACTICE — almost always mini-batch
# PyTorch DataLoader returns mini-batches by default
```

### 3.3 Learning Rate: The Most Important Hyperparameter

```
Learning rate (η) controls step size:

Too small (η = 0.0001):  ........→ slowly → takes forever to converge
Just right (η = 0.01):   ...→    converges nicely
Too large (η = 1.0):     →→→→EXPLODES! ← diverges, loss goes to infinity

How to find the right LR:
1. Start with 0.001 or 0.01
2. If loss doesn't decrease → increase LR
3. If loss is unstable/NaN → decrease LR
4. Use learning rate schedulers (decay over time)
```

### 3.4 Advanced Optimisers

**Momentum:**
```python
# Problem: SGD oscillates in narrow valleys (zig-zag path)
# Solution: Add a "velocity" term — like a ball rolling downhill
v = 0
for batch in batches:
    gradient = compute_gradient(batch)
    v = momentum * v + gradient          # accumulate past gradients
    weights -= lr * v                     # update with velocity

# momentum = 0.9 is standard
# Effect: smooths out oscillations, accelerates along consistent gradients
```

**Adam (Adaptive Moment Estimation) — the default optimiser:**
```python
# Combines Momentum + adaptive learning rates per parameter
# Uses first moment (mean) and second moment (variance) of gradients

m = 0   # first moment (momentum-like)
v = 0   # second moment (RMSprop-like)
t = 0

for batch in batches:
    t += 1
    gradient = compute_gradient(batch)
    
    m = beta1 * m + (1 - beta1) * gradient           # momentum
    v = beta2 * v + (1 - beta2) * gradient**2         # RMSprop
    
    m_hat = m / (1 - beta1**t)  # bias correction (early steps are biased toward 0)
    v_hat = v / (1 - beta2**t)
    
    weights -= lr * m_hat / (sqrt(v_hat) + epsilon)   # adaptive step

# Defaults: beta1=0.9, beta2=0.999, epsilon=1e-8, lr=0.001
# Why it works: rare features get larger updates, frequent features get smaller
# Used by: almost every deep learning paper, PyTorch default
```

**AdamW (Adam with decoupled weight decay):**
```python
# Adam has a subtle issue: L2 regularisation doesn't work correctly
# because Adam adapts the effective regularisation strength per parameter
# AdamW fixes this by decoupling weight decay from the gradient update

weights -= lr * m_hat / (sqrt(v_hat) + epsilon) + lr * weight_decay * weights

# Used by: all modern transformer training (GPT, BERT, LLaMA)
# This is what Hugging Face Transformers uses by default
```

### 3.5 Learning Rate Schedulers

```python
# Constant LR is rarely optimal. Common schedules:

# 1. Step decay: reduce LR by factor every N epochs
scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
# LR: 0.01 → 0.001 → 0.0001 at epochs 30, 60, 90

# 2. Cosine annealing: smooth cosine decrease
scheduler = CosineAnnealingLR(optimizer, T_max=100)
# LR follows a cosine curve from initial to 0

# 3. Warmup + cosine (most common for transformers):
# Start with tiny LR, linearly increase for N steps, then cosine decay
# Warmup prevents early gradient explosions when weights are random
# Used by: BERT, GPT-3, LLaMA, all modern LLM training

# 4. One Cycle (fast convergence):
scheduler = OneCycleLR(optimizer, max_lr=0.01, total_steps=1000)
```

### 3.6 Why Gradient Descent Works for Neural Networks

```
Neural network loss landscape:
  - Not convex (many local minima)
  - High-dimensional (millions of parameters)
  
So why does gradient descent work?
1. Most local minima are nearly as good as global minimum (empirical finding)
2. Saddle points are more dangerous than local minima — momentum helps escape them
3. SGD noise actually helps explore the landscape
4. Overparameterised networks (more params than data points) have smoother landscapes

Key insight: the real challenge is not finding THE minimum,
but finding A minimum that generalises well to unseen data.
That's where regularisation comes in...
```

> 🃏 **Quick-Recall Card — Gradient Descent**
> | Concept | One-liner |
> |---|---|
> | Gradient | Direction of steepest ascent of loss. We go the opposite way. |
> | Learning Rate (η) | Step size. Too big = explode. Too small = crawls forever. |
> | Batch GD | All data per update. Stable but slow on big datasets. |
> | SGD | 1 sample per update. Fast & noisy. Can escape local minima. |
> | Mini-Batch | B samples (32–256). Best of both worlds. **What's used in practice.** |
> | Adam | SGD + momentum + per-param adaptive LR. **Default choice for neural nets.** |
> | Warmup+Cosine | Ramp LR up, then decay. Used in every modern LLM training run. |
> | Gradient Clipping | Cap gradient norm (1.0) to prevent NaN explosions in Transformers. |

---

<a name="part-4"></a>
## Part 4 — Logistic Regression

### 4.1 From Linear to Logistic

Linear regression outputs any number (−∞ to +∞). For **binary classification** (yes/no, spam/not-spam), we need output between 0 and 1 (a probability).

**Solution:** Apply the **sigmoid function** to the linear output:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

```python
import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Properties:
# sigmoid(0) = 0.5
# sigmoid(large positive) → 1.0
# sigmoid(large negative) → 0.0
# Always between 0 and 1 ✓
```

```
sigmoid output:
  1.0 ─────────────────────────/
                              /
  0.5 ───────────────────────/ ─
                            /
  0.0 ─────────────────────/
         -6  -4  -2   0   2   4   6  → z
```

**The model:**
```
z = w·x + b            (linear part — same as linear regression)
ŷ = σ(z) = 1/(1+e⁻ᶻ)  (apply sigmoid to get probability)

Prediction:
  if ŷ ≥ 0.5: class 1 (positive)
  if ŷ < 0.5: class 0 (negative)
```

### 4.2 The Loss Function: Binary Cross-Entropy

MSE doesn't work well for classification (loss surface is non-convex with sigmoid → many bad local minima). Instead, use **Binary Cross-Entropy (BCE)**:

$$\text{BCE} = -\frac{1}{n} \sum_{i=1}^n \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

**Intuition:**
```
If true label y=1 (positive):
  loss = -log(ŷ)
  If ŷ → 1.0: loss → 0    (correct, small loss ✓)
  If ŷ → 0.0: loss → ∞    (wrong, huge loss ✗ — strong penalty!)

If true label y=0 (negative):
  loss = -log(1-ŷ)
  If ŷ → 0.0: loss → 0    (correct, small loss ✓)
  If ŷ → 1.0: loss → ∞    (wrong, huge loss ✗)
```

### 4.3 Logistic Regression from Scratch

```python
class LogisticRegression:
    def __init__(self, lr=0.01, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
    
    def sigmoid(self, z):
        # Clip to prevent overflow
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.losses = []
        
        for _ in range(self.n_iter):
            # Forward pass
            z = X @ self.weights + self.bias
            y_pred = self.sigmoid(z)
            
            # BCE loss
            epsilon = 1e-7  # prevent log(0)
            loss = -np.mean(y * np.log(y_pred + epsilon) + 
                           (1 - y) * np.log(1 - y_pred + epsilon))
            self.losses.append(loss)
            
            # Gradients (derived from BCE loss)
            dw = (1 / n_samples) * X.T @ (y_pred - y)
            db = (1 / n_samples) * np.sum(y_pred - y)
            
            # Update
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
    
    def predict_proba(self, X):
        return self.sigmoid(X @ self.weights + self.bias)
    
    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)
```

### 4.4 Multi-Class: Softmax Regression

For K classes (not just 2), replace sigmoid with **softmax**:

$$\text{softmax}(z_j) = \frac{e^{z_j}}{\sum_{k=1}^K e^{z_k}}$$

```python
def softmax(z):
    # z shape: (n_samples, K classes)
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))  # subtract max for numerical stability
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)

# Each output is a probability, all K outputs sum to 1.0
# Loss: Categorical Cross-Entropy
# This is exactly what LLMs use in their final layer!
```

### 4.5 Decision Boundary

The decision boundary of logistic regression is a **hyperplane** (linear boundary):

```
2D example:
  w₁x₁ + w₂x₂ + b = 0  → a straight line

Points on one side: class 1
Points on other side: class 0

If data is linearly separable, logistic regression works well.
If not (e.g., XOR problem), you need a non-linear model (SVM with kernel, neural network, tree).
```

> 🃏 **Quick-Recall Card — Logistic Regression**
> | Concept | One-liner |
> |---|---|
> | Sigmoid | Squashes any value to (0,1). Gives probability for binary class. |
> | BCE Loss | −log(ŷ) for label=1. Exponential penalty for confident wrong predictions. |
> | Decision boundary | A straight line (hyperplane). Assumes linearly separable data. |
> | Softmax | Multi-class version. K outputs summing to 1.0. Used in every LLM output layer. |
> | Gradient | (ŷ − y) × x — elegantly same formula regardless of sigmoid or softmax. |

---

<a name="part-5"></a>
## Part 5 — The Bias-Variance Tradeoff

> **This is the #1 most-asked ML fundamentals question in AI engineer interviews.**

> 💡 **ELI5 (Explain Like I'm 5):** 
> Imagine studying for a test by taking practice exams. 
> * **High Bias (Underfitting):** You don't study at all. You just write "C" for every multiple choice question. You fail the practice test and the real test.
> * **High Variance (Overfitting):** You memorise the exact answers (A, B, D, A) to the practice test without understanding the material. You get 100% on the practice test, but fail the real test because the questions are slightly different.
> * **The Sweet Spot:** You understand the core concepts. You do decently on the practice test, and similarly well on the real test.

### 5.1 What Is Bias?

**Bias** = how much the model's average prediction differs from the true value (across different training sets).

```
High bias → model is too simple → underfitting
  - The model can't capture the true pattern
  - Example: fitting a straight line to curved data
  - Training error is high and test error is high
  - "The model is wrong even on training data"
```

### 5.2 What Is Variance?

**Variance** = how much the model's predictions change when trained on different training sets.

```
High variance → model is too complex → overfitting
  - The model memorises noise in the training data
  - Example: fitting a degree-20 polynomial to 10 points
  - Training error is LOW but test error is HIGH
  - "The model learned the training data perfectly but fails on new data"
```

### 5.3 The Decomposition

$$\text{Expected Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$$

```
Irreducible noise: randomness in the data itself (sensor noise, measurement error)
  → Cannot be reduced by any model
  → Sets the floor for achievable error

Total error = Bias² + Variance + Noise
```

### 5.4 The Visual Tradeoff

```
Error
  |
  |  \                        /     ← Test error (U-shaped)
  |   \      _______________/
  |    \   /
  |     \_/     ← Sweet spot
  |                  
  |     ________________________  ← Training error (always decreases)
  |
  +─────────────────────────────→ Model complexity
  
  Simple                     Complex
  (high bias,               (low bias,
   low variance)             high variance)
```

### 5.5 Concrete Examples

| Model | Bias | Variance | Risk |
|---|---|---|---|
| Linear regression (few features) | High | Low | Underfitting |
| Decision tree (max depth) | Low | Very high | Overfitting |
| Random Forest | Low | Medium | Good balance |
| k-NN with k=1 | Zero (on training set) | Very high | Overfitting |
| k-NN with k=n | Very high | Zero | Underfitting |
| Neural network (large) | Low | High* | *Regularise! |

### 5.6 How to Diagnose

```
Training error LOW, Test error LOW    → Good fit ✓
Training error HIGH, Test error HIGH  → Underfitting (high bias)
  → Fix: more features, more complex model, fewer regularisation
  
Training error LOW, Test error HIGH   → Overfitting (high variance)
  → Fix: more training data, regularisation, simpler model, dropout, early stopping
  
Training error HIGH, Test error HIGHER → Both (rare, usually underfitting)
```

### 5.7 The Interview Answer

> "The bias-variance tradeoff states that a model's total error is the sum of bias squared, variance, and irreducible noise. Bias measures systematic error from overly simple assumptions — high bias means underfitting. Variance measures sensitivity to training data fluctuations — high variance means overfitting. We can't minimise both simultaneously: reducing one tends to increase the other. The goal is to find the sweet spot using techniques like cross-validation, regularisation, and ensemble methods."

> 🃏 **Quick-Recall Card — Bias-Variance Tradeoff**
> | Scenario | Signal | Fix |
> |---|---|---|
> | Train error HIGH, Test error HIGH | High Bias (Underfitting) | More features, deeper model, less regularisation |
> | Train error LOW, Test error HIGH | High Variance (Overfitting) | More data, regularisation, dropout, simpler model |
> | Train error LOW, Test error LOW | Good fit ✓ | Ship it. |
> | Both errors HIGH | Underfitting AND noisy data | Fix data quality first |
>
> **One-sentence interview answer:** *"Bias is systematic error from overly simple assumptions; variance is sensitivity to training data. The tradeoff: reducing one tends to increase the other — use cross-validation, regularisation, and ensemble methods to find the sweet spot."*

---

<a name="part-6"></a>
## Part 6 — Regularisation

### 6.1 Why Regularisation?

Regularisation **penalises model complexity** to prevent overfitting. It adds a penalty term to the loss function:

$$\text{Regularised Loss} = \text{Original Loss} + \lambda \times \text{Penalty}$$

Where $\lambda$ (lambda) controls the strength of regularisation.

### 6.2 L1 Regularisation (Lasso)

$$\text{Penalty} = \lambda \sum_{j=1}^p \lvert w_j \rvert$$

```
Effect: Drives some weights to EXACTLY zero → automatic feature selection
Use when: You suspect many features are irrelevant
Result: Sparse model (few non-zero weights)

Example: 100 features → L1 might keep only 15 with non-zero weights
The other 85 are effectively removed from the model
```

### 6.3 L2 Regularisation (Ridge)

$$\text{Penalty} = \lambda \sum_{j=1}^p w_j^2$$

```
Effect: Shrinks ALL weights toward zero (but none exactly to zero)
Use when: You want to keep all features but prevent any one from dominating
Result: Small, evenly distributed weights

L2 penalises large weights QUADRATICALLY:
  weight = 10 → penalty = 100
  weight = 1  → penalty = 1
  This strongly discourages large weights
```

### 6.4 ElasticNet (L1 + L2)

$$\text{Penalty} = \alpha \lambda \sum \lvert w_j \rvert + \frac{(1-\alpha)}{2} \lambda \sum w_j^2$$

Best of both worlds: feature selection (L1) + weight shrinkage (L2).

### 6.5 Implementation with Scikit-Learn

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.model_selection import cross_val_score

# No regularisation
lr = LinearRegression()

# L2 (Ridge) — alpha = lambda
ridge = Ridge(alpha=1.0)  # higher alpha = stronger regularisation

# L1 (Lasso)
lasso = Lasso(alpha=0.1)

# ElasticNet
elastic = ElasticNet(alpha=0.1, l1_ratio=0.5)  # l1_ratio: balance between L1 and L2

# Compare with cross-validation
for model, name in [(lr, "LinReg"), (ridge, "Ridge"), (lasso, "Lasso")]:
    scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
    print(f"{name}: MSE = {-scores.mean():.4f} ± {scores.std():.4f}")
```

### 6.6 Regularisation for Neural Networks

**Dropout:**
```python
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(100, 64)
        self.dropout = nn.Dropout(p=0.3)  # randomly zero 30% of activations
        self.fc2 = nn.Linear(64, 1)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)   # ONLY during training! Disabled at eval()
        x = self.fc2(x)
        return x

# How dropout works:
# Training: each neuron has p=0.3 chance of being "turned off" each forward pass
# This forces the network to be robust — can't rely on any single neuron
# Effect: like training an ensemble of sub-networks
# Inference: all neurons active, weights scaled by (1-p) to compensate
```

**Early Stopping:**
```python
# Stop training when validation loss stops improving
best_val_loss = float('inf')
patience = 10
no_improve_count = 0

for epoch in range(max_epochs):
    train_loss = train_one_epoch()
    val_loss = evaluate(val_data)
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_model()
        no_improve_count = 0
    else:
        no_improve_count += 1
        if no_improve_count >= patience:
            print(f"Early stopping at epoch {epoch}")
            break
```

**Weight Decay (L2 for neural networks):**
```python
# PyTorch: add weight_decay to optimiser
optimiser = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
# This adds L2 penalty to all parameter updates
```

**Data Augmentation (for images):**
```
Regulariser for free: create more training data by transforming existing samples
  - Flip, rotate, crop, colour jitter, scale
  - The model sees more variations → generalises better
  - Effect: reduces variance without changing model architecture
```

### 6.7 Summary: When to Use What

| Technique | Best For | Reduces |
|---|---|---|
| L1 (Lasso) | Feature selection, sparse models | Variance |
| L2 (Ridge) | General-purpose, keeps all features | Variance |
| Dropout | Neural networks | Variance |
| Early stopping | Any iterative model | Variance |
| Weight decay | Transformers, deep nets | Variance |
| Data augmentation | Image models | Variance |
| Batch normalisation | Deep networks | Both (stabilises training) |

---

<a name="part-7"></a>
## Part 7 — Evaluation Metrics: The Complete Guide

> **Critical for interviews.** You MUST know when to use each metric and WHY. "I'd use accuracy" is often the wrong answer.

### 7.1 The Confusion Matrix

```
                    Predicted
                  Pos      Neg
Actual  Pos  [   TP    |   FN   ]
        Neg  [   FP    |   TN   ]

TP (True Positive):  Model said YES, actually YES ✓
FP (False Positive): Model said YES, actually NO  ✗ (Type I error)
FN (False Negative): Model said NO,  actually YES ✗ (Type II error)
TN (True Negative):  Model said NO,  actually NO  ✓
```

### 7.2 Accuracy

$$\text{Accuracy} = \frac{TP + TN}{TP + FP + FN + TN}$$

```
"What percentage of all predictions were correct?"

Problem: USELESS for imbalanced data!
  Example: 99% of emails are not spam, 1% are spam
  A model that ALWAYS predicts "not spam" gets 99% accuracy!
  But it catches zero spam — completely useless

Rule: Only use accuracy when classes are roughly balanced (50/50)
```

### 7.3 Precision

$$\text{Precision} = \frac{TP}{TP + FP}$$

```
"Of all items the model PREDICTED as positive, how many actually were?"

High precision = few false positives
"When it says YES, it's usually right"

When precision matters most:
  - Spam filter: falsely marking a real email as spam is very costly
  - Criminal sentencing: falsely convicting an innocent person
  - Recommender: suggesting irrelevant items annoys users
```

### 7.4 Recall (Sensitivity, True Positive Rate)

$$\text{Recall} = \frac{TP}{TP + FN}$$

```
"Of all items that ARE actually positive, how many did the model find?"

High recall = few missed positives
"It catches almost everything important"

When recall matters most:
  - Cancer screening: missing a cancer case is life-threatening
  - Fraud detection: missing a fraud costs the bank money
  - Search: returning all relevant results is important
```

### 7.5 F1 Score (Harmonic Mean of Precision and Recall)

$$F_1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}$$

```
F1 balances precision and recall: only high if BOTH are high.
  Precision=1.0, Recall=0.01 → F1 = 0.02  (terrible recall ruins it)
  Precision=0.8, Recall=0.8  → F1 = 0.8   (balanced, good)

When F1 matters:
  - Imbalanced datasets where accuracy is misleading
  - When you need a single number to compare models
  - NLP tasks (text classification, NER)
```

### 7.6 Precision-Recall Tradeoff

```
By changing the classification threshold (default 0.5):
  - Higher threshold (e.g., 0.9): fewer positives predicted
    → Precision ↑ (only very confident predictions)
    → Recall ↓ (many actual positives missed)
    
  - Lower threshold (e.g., 0.1): more positives predicted
    → Precision ↓ (many false positives)
    → Recall ↑ (catches almost everything)

You CHOOSE the threshold based on the business requirement:
  Cancer screening: set threshold LOW (catch everything, even at cost of false alarms)
  Spam filter: set threshold HIGH (don't incorrectly block real emails)
```

### 7.7 AUC-ROC (Area Under the ROC Curve)

```
ROC Curve:
  X-axis: False Positive Rate (FPR) = FP / (FP + TN)
  Y-axis: True Positive Rate (TPR) = TP / (TP + FN) = Recall
  
  Plot TPR vs FPR at every threshold from 0 to 1
  
AUC = Area Under this curve
  AUC = 1.0: perfect classifier
  AUC = 0.5: random guessing (diagonal line)
  AUC < 0.5: worse than random (flip predictions!)

When to use AUC-ROC:
  - Comparing models across ALL thresholds (not just 0.5)
  - Binary classification benchmark
  - When the threshold hasn't been decided yet
  
When NOT to use AUC-ROC:
  - Heavily imbalanced data → use AUC-PR instead
  - (AUC-ROC can look good even if the model fails on rare class)
```

### 7.8 AUC-PR (Precision-Recall AUC)

```
PR Curve:
  X-axis: Recall
  Y-axis: Precision
  
Better for imbalanced datasets because:
  - Focuses on the positive (minority) class performance
  - Not inflated by the large number of true negatives
  
Example: 1% fraud rate
  AUC-ROC might be 0.98 just from getting negatives right
  AUC-PR might be 0.45 showing the model struggles on actual fraud cases
```

### 7.9 Regression Metrics

```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# MSE: average squared error (penalises large errors)
mse = mean_squared_error(y_true, y_pred)

# RMSE: square root of MSE (same units as target)
rmse = np.sqrt(mse)

# MAE: average absolute error (robust to outliers)
mae = mean_absolute_error(y_true, y_pred)

# R² (coefficient of determination)
# "What fraction of the variance in y does the model explain?"
# R² = 1 - (sum of squared residuals / total variance)
# R² = 1.0: perfect model
# R² = 0.0: model is no better than predicting the mean
# R² < 0.0: model is WORSE than predicting the mean
r2 = r2_score(y_true, y_pred)
```

### 7.10 Complete Metrics Code

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, classification_report,
    confusion_matrix
)

y_true = [0, 0, 1, 1, 0, 1, 0, 1, 1, 0]
y_pred = [0, 1, 1, 1, 0, 0, 0, 1, 1, 0]

print(f"Accuracy:  {accuracy_score(y_true, y_pred):.3f}")
print(f"Precision: {precision_score(y_true, y_pred):.3f}")
print(f"Recall:    {recall_score(y_true, y_pred):.3f}")
print(f"F1:        {f1_score(y_true, y_pred):.3f}")

print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=["Negative", "Positive"]))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
```

---

<a name="part-8"></a>
## Part 8 — Cross-Validation

### 8.1 Why Not Just Train/Test Split?

```
Problem:
  If you split 80/20, the test score depends on WHICH 20% you picked.
  You might get lucky (easy test examples) or unlucky (hard ones).
  A single split gives unreliable performance estimates.

Solution: K-Fold Cross-Validation
  Split data into K equal parts (folds)
  Train K times, each time using a different fold as "test" and rest as "train"
  Average all K test scores → much more reliable estimate
```

### 8.2 K-Fold Cross-Validation

```python
from sklearn.model_selection import cross_val_score, KFold

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# 5 separate train/test cycles:
# Fold 1: [TEST] [TRAIN] [TRAIN] [TRAIN] [TRAIN]
# Fold 2: [TRAIN] [TEST] [TRAIN] [TRAIN] [TRAIN]
# Fold 3: [TRAIN] [TRAIN] [TEST] [TRAIN] [TRAIN]
# Fold 4: [TRAIN] [TRAIN] [TRAIN] [TEST] [TRAIN]
# Fold 5: [TRAIN] [TRAIN] [TRAIN] [TRAIN] [TEST]

scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
print(f"Accuracy: {scores.mean():.3f} ± {scores.std():.3f}")
# The ± tells you how stable the model is
# High std → model performance depends heavily on which data it sees (variance!)
```

### 8.3 Stratified K-Fold (For Classification)

```python
from sklearn.model_selection import StratifiedKFold

# Problem: If class is imbalanced (95% negative, 5% positive)
# A random fold might get 0 positive examples → useless fold

# StratifiedKFold ensures each fold has same class proportion as full dataset
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf, scoring='f1')
```

### 8.4 Time Series Cross-Validation

```python
from sklearn.model_selection import TimeSeriesSplit

# Problem: for time-series data, you can't randomly shuffle!
# Training must use PAST data to predict FUTURE data.

# TimeSeriesSplit:
# Fold 1: [TRAIN] [TEST]
# Fold 2: [TRAIN   TRAIN] [TEST]
# Fold 3: [TRAIN   TRAIN   TRAIN] [TEST]
# Each fold: train on all past, test on immediate future

tscv = TimeSeriesSplit(n_splits=5)
scores = cross_val_score(model, X_time, y_time, cv=tscv)
```

### 8.5 Nested Cross-Validation (Hyperparameter Tuning)

```python
from sklearn.model_selection import GridSearchCV

# Common mistake: tune hyperparameters on test set → overly optimistic performance!
# Correct approach: use validation set for tuning, test set only for final evaluation

# Option 1: train/validation/test split (60/20/20)
# Option 2: Nested CV (outer CV for evaluation, inner CV for tuning)

param_grid = {'C': [0.01, 0.1, 1, 10, 100]}
inner_cv = StratifiedKFold(n_splits=3)
outer_cv = StratifiedKFold(n_splits=5)

grid = GridSearchCV(SVC(), param_grid, cv=inner_cv, scoring='f1')
nested_scores = cross_val_score(grid, X, y, cv=outer_cv, scoring='f1')
print(f"Nested CV F1: {nested_scores.mean():.3f} ± {nested_scores.std():.3f}")
```

---

<a name="part-9"></a>
## Part 9 — Decision Trees

### 9.1 How Decision Trees Work

A decision tree recursively splits data based on feature values to make predictions.

```
                    [Is income > $50K?]
                    /               \
                 Yes                 No
              /                        \
    [Age > 35?]                   [Has degree?]
     /       \                     /         \
   Yes       No                  Yes          No
    ↓         ↓                   ↓            ↓
  Approve   Review             Approve       Deny
```

Each **internal node** = a question about a feature
Each **leaf node** = a prediction (class for classification, mean value for regression)
Each **branch** = an outcome of the question

### 9.2 How Splits Are Chosen: Gini Impurity

The tree tries every possible feature and every possible threshold to find the split that produces the **purest** child nodes.

**Gini Impurity:** Probability of misclassifying a randomly chosen element.

$$\text{Gini}(S) = 1 - \sum_{k=1}^K p_k^2$$

```
Example:
  Node A: 50% class 0, 50% class 1 → Gini = 1 - (0.5² + 0.5²) = 0.5 (max impurity)
  Node B: 90% class 0, 10% class 1 → Gini = 1 - (0.9² + 0.1²) = 0.18 (fairly pure)
  Node C: 100% class 0             → Gini = 1 - (1.0²) = 0.0 (perfectly pure)
  
Best split = the split that reduces weighted Gini the most
```

### 9.3 Information Gain (Entropy-Based)

Alternative to Gini: use **entropy** (from information theory).

$$\text{Entropy}(S) = -\sum_{k=1}^K p_k \log_2(p_k)$$

$$\text{Information Gain} = \text{Entropy(parent)} - \sum \frac{\lvert S_i \rvert}{\lvert S \rvert} \text{Entropy}(S_i)$$

```
Entropy:
  50/50 split → Entropy = -0.5·log₂(0.5) - 0.5·log₂(0.5) = 1.0 bit (max uncertainty)
  90/10 split → Entropy = -0.9·log₂(0.9) - 0.1·log₂(0.1) ≈ 0.47 bits
  100/0 split → Entropy = 0.0 bits (completely certain)

Gini and Entropy usually produce the same splits. Gini is slightly faster (no log).
Scikit-learn default: Gini.
```

### 9.4 Decision Tree Implementation

```python
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)

# Key hyperparameters for controlling overfitting:
tree = DecisionTreeClassifier(
    criterion='gini',       # or 'entropy'
    max_depth=5,            # limit tree depth (most important!)
    min_samples_split=10,   # min samples to consider splitting a node
    min_samples_leaf=5,     # min samples in each leaf
    max_features='sqrt',    # consider sqrt(n_features) per split (randomisation)
    random_state=42
)

tree.fit(X, y)
print(export_text(tree, feature_names=load_iris().feature_names))
```

### 9.5 Decision Tree Pros and Cons

```
Pros:
  ✓ Interpretable (can visualise and explain decisions)
  ✓ No feature scaling needed (splits don't depend on absolute values)
  ✓ Handles both numerical and categorical features
  ✓ Fast inference (just follow the branches)
  ✓ Non-linear decision boundaries

Cons:
  ✗ HIGH VARIANCE — small data changes → completely different tree (unstable)
  ✗ Prone to overfitting (without regularisation, it memorises training data)
  ✗ Greedy algorithm (locally optimal splits, not globally optimal)
  ✗ Biased toward features with many values (more split options)

Fix: → Ensemble methods (Random Forest, XGBoost) solve all these cons!
```

---

<a name="part-10"></a>
## Part 10 — Ensemble Methods

### 10.1 The Core Idea

**Combining multiple weak models into one strong model.** Like asking 100 people a question and going with the majority vote — more reliable than asking one expert.

### 10.2 Bagging (Bootstrap Aggregating)

```
1. Create B bootstrap samples (random sampling WITH replacement)
2. Train a separate model on each bootstrap sample
3. Aggregate predictions:
   - Classification: majority vote
   - Regression: average

Sample 1: [x₃, x₁, x₁, x₅, x₂] → Model 1 → predict A
Sample 2: [x₂, x₄, x₃, x₃, x₁] → Model 2 → predict B
Sample 3: [x₅, x₁, x₂, x₄, x₄] → Model 3 → predict A
                                                    → Final: A (majority)

Why it works: Each model sees different data, so they make DIFFERENT errors.
Averaging cancels out individual errors → reduces VARIANCE
```

### 10.3 Random Forest = Bagging + Random Feature Selection

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,       # number of trees (more = better, diminishing returns)
    max_depth=10,           # limit each tree's depth
    max_features='sqrt',    # each split considers only √(n_features) random features
    min_samples_leaf=5,
    n_jobs=-1,              # use all CPU cores
    random_state=42
)

rf.fit(X_train, y_train)
print(f"Accuracy: {rf.score(X_test, y_test):.3f}")

# Feature importance
for name, importance in sorted(zip(feature_names, rf.feature_importances_), 
                               key=lambda x: -x[1]):
    print(f"  {name}: {importance:.4f}")
```

**Why Random Forest is so good:**
```
Decision Tree: high variance (unstable, overfits)
Random Forest: low variance (stable, generalises)

Two sources of randomness:
1. Bootstrap sampling: each tree trains on a random subset of examples
2. Feature subsampling: each split considers a random subset of features

This decorrelates the trees — they make independent errors that cancel out.
Result: almost always better than a single decision tree. Rarely needs tuning.
```

### 10.4 Boosting: Building Models Sequentially

```
Unlike bagging (parallel models), boosting builds models SEQUENTIALLY:
  
Model 1: fit to data → make errors on some examples
Model 2: focus MORE on the examples Model 1 got wrong
Model 3: focus MORE on the examples Models 1+2 still get wrong
...
Final: weighted combination of all models

Key idea: each new model CORRECTS the errors of previous models
Result: reduces BIAS (can learn complex patterns)
```

### 10.5 XGBoost (eXtreme Gradient Boosting)

XGBoost is the #1 algorithm for structured/tabular data (Kaggle competitions, production ML).

```python
import xgboost as xgb
from sklearn.model_selection import cross_val_score

model = xgb.XGBClassifier(
    n_estimators=300,           # number of boosting rounds
    max_depth=6,                # depth per tree (lower = less overfitting)
    learning_rate=0.1,          # shrinkage (lower = more rounds needed, better)
    subsample=0.8,              # fraction of samples per tree (like bagging)
    colsample_bytree=0.8,      # fraction of features per tree
    reg_alpha=0.1,              # L1 regularisation
    reg_lambda=1.0,             # L2 regularisation
    eval_metric='logloss',
    early_stopping_rounds=20,   # stop if validation doesn't improve
    random_state=42
)

# Train with early stopping
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

print(f"Best iteration: {model.best_iteration}")
print(f"Test accuracy: {model.score(X_test, y_test):.3f}")
```

**Why XGBoost is special:**
```
1. Gradient boosting: each tree fits the GRADIENT of the loss (residuals)
2. Regularisation built-in: L1 + L2 on leaf weights
3. Column subsampling: like random forest's feature randomisation
4. Handles missing values: learns optimal direction for missing
5. Parallelised tree building: fast despite being sequential
6. Pruning: trees are pruned using a maximum loss reduction threshold
```

### 10.6 LightGBM

```python
import lightgbm as lgb

model = lgb.LGBMClassifier(
    n_estimators=300,
    max_depth=-1,               # -1 = no limit
    num_leaves=31,              # max leaves per tree (controls complexity)
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42
)

# LightGBM grows trees LEAF-WISE (best-first) instead of LEVEL-WISE
# This usually reaches better accuracy faster but can overfit more
# Use num_leaves to control complexity instead of max_depth
```

### 10.7 Bagging vs Boosting Summary

| | Bagging (Random Forest) | Boosting (XGBoost) |
|---|---|---|
| How | Parallel, independent models | Sequential, dependent models |
| Reduces | Variance | Bias (and variance) |
| Base learner | Deep, complex trees | Shallow, simple trees |
| Overfitting risk | Low | Higher (needs early stopping) |
| Tuning effort | Minimal | More hyperparameters to tune |
| Best for | Quick, reliable baseline | Max accuracy with tuning |

---

<a name="part-11"></a>
## Part 11 — Support Vector Machines (SVM)

### 11.1 The Idea

Find the **hyperplane** that maximally separates two classes. The data points closest to the hyperplane are called **support vectors** — they define the boundary.

```
         ·  ·                Best hyperplane:
        ·  ·   |   o  o     maximises the MARGIN
       ·  · ← SV  | ← SV → o  o    (distance between hyperplane
        ·  ·   |   o  o     and nearest points on each side)
         ·  ·      o
         
    Class -1  margin  Class +1
```

### 11.2 The Kernel Trick

When data isn't linearly separable, project it to a higher-dimensional space where it IS.

```
Original 1D: · · · o o o · · · (can't separate with a line)
Project to 2D by adding x² feature:
    ↑
    · · ·           
            o o o ← now separable by a horizontal line
    · · ·
    →

The kernel trick computes dot products in the high-dimensional space
WITHOUT actually transforming the data → efficient!
```

```python
from sklearn.svm import SVC

# Linear kernel: simple hyperplane
svm_linear = SVC(kernel='linear', C=1.0)

# RBF (Radial Basis Function) kernel: non-linear boundary
svm_rbf = SVC(kernel='rbf', C=1.0, gamma='scale')
# gamma: how far each training example's influence reaches
# small gamma → smooth boundary (risk: underfitting)
# large gamma → tight boundary (risk: overfitting)

# C: regularisation parameter (trade-off)
# small C → wider margin, more misclassifications allowed (soft margin)
# large C → narrow margin, fewer misclassifications (harder margin, risk overfit)
```

### 11.3 When to Use SVM

```
Good for:
  ✓ Small to medium datasets (< 100K samples)
  ✓ High-dimensional data (works well even with more features than samples)
  ✓ Text classification (with TF-IDF features)
  ✓ When you need a strong non-linear model without ensemble complexity

Not good for:
  ✗ Large datasets (training is O(n² to n³) — slow for >100K samples)
  ✗ When you need probability outputs (SVM gives hard decisions by default)
  ✗ When interpretability is required
```

---

<a name="part-12"></a>
## Part 12 — K-Nearest Neighbours (KNN)

### 12.1 The Simplest Algorithm

No training at all. At prediction time, find the K closest training examples and vote.

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(
    n_neighbors=5,          # K: number of neighbours to consider
    metric='euclidean',     # distance metric
    weights='uniform'       # 'uniform' (majority vote) or 'distance' (closer = more weight)
)
knn.fit(X_train, y_train)    # just stores the data!
y_pred = knn.predict(X_test) # O(n) per prediction — must scan all training points

# Choosing K:
# K=1: perfectly matches training data (zero training error)
#       but highly sensitive to noise → overfitting (high variance)
# K=large (e.g., K=100): very smooth predictions
#       but may miss local patterns → underfitting (high bias)
# Rule of thumb: K = √n, then tune with cross-validation
# Always use ODD K for binary classification (avoid ties)
```

### 12.2 The Curse of Dimensionality

```
KNN breaks in high dimensions:
  In 1D: K nearest neighbours are genuinely "near"
  In 100D: all points are roughly equidistant (distances converge)
  In 1000D: the concept of "nearest" becomes meaningless

Fix: dimensionality reduction (PCA) before KNN
Fix: use tree-based models or neural networks instead
```

---

<a name="part-13"></a>
## Part 13 — Naive Bayes

### 13.1 Bayes' Theorem

$$P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}$$

```
In classification terms:
  P(class | features) = P(features | class) × P(class) / P(features)

P(spam | "free money click") = P("free money click" | spam) × P(spam) / P("free money click")
```

### 13.2 The "Naive" Assumption

Assumes all features are **conditionally independent** given the class:

$$P(x_1, x_2, \ldots, x_n \mid y) = \prod_{i=1}^n P(x_i \mid y)$$

```
This is almost never true in real data, but it works surprisingly well!
Why? Because classification only needs the RANKING of probabilities to be correct,
not the actual probability values. And the ranking is often preserved.
```

### 13.3 Types of Naive Bayes

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# Gaussian: continuous features (assumes normal distribution)
gnb = GaussianNB()

# Multinomial: count features (word counts in text classification)
mnb = MultinomialNB(alpha=1.0)  # alpha = Laplace smoothing (prevents P=0)

# Bernoulli: binary features (word present/absent)
bnb = BernoulliNB(alpha=1.0)

# Classic use case: text classification (spam detection)
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import make_pipeline

text_clf = make_pipeline(
    CountVectorizer(),
    MultinomialNB()
)
text_clf.fit(X_train_text, y_train)
```

### 13.4 Pros and Cons

```
Pros:
  ✓ Extremely fast training and prediction (even faster than logistic regression)
  ✓ Works well with high-dimensional data (text with 50K+ features)
  ✓ Needs very little training data
  ✓ Good baseline for text classification

Cons:
  ✗ Independence assumption rarely holds → calibrated probabilities are poor
  ✗ Continuous features: assumes Gaussian distribution (often wrong)
  ✗ Cannot learn interactions between features
```

---

<a name="part-14"></a>
## Part 14 — Clustering

### 14.1 K-Means Clustering

```python
from sklearn.cluster import KMeans

# Algorithm:
# 1. Randomly place K centroids
# 2. Assign each point to nearest centroid → K clusters
# 3. Recompute centroids as mean of assigned points
# 4. Repeat until centroids stop moving (convergence)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

# Choosing K: Elbow Method
inertias = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X)
    inertias.append(km.inertia_)  # sum of squared distances to closest centroid

# Plot inertias vs K → look for "elbow" (point of diminishing returns)
```

**Limitations of K-Means:**
```
✗ Must specify K in advance
✗ Assumes spherical clusters (equal-sized, circular shape)
✗ Sensitive to initialisation → run multiple times (n_init>1)
✗ Cannot find non-convex clusters
```

### 14.2 DBSCAN (Density-Based Clustering)

```python
from sklearn.cluster import DBSCAN

# No need to specify K!
# Finds clusters as dense regions separated by sparse regions

dbscan = DBSCAN(
    eps=0.5,          # maximum distance between two neighbours
    min_samples=5     # minimum points to form a dense region
)
labels = dbscan.fit_predict(X)
# labels = -1 means noise (outlier, not in any cluster)

# Pros: no K needed, finds arbitrary-shaped clusters, handles outliers
# Cons: sensitive to eps/min_samples, struggles with varying density
```

---

<a name="part-15"></a>
## Part 15 — Dimensionality Reduction

### 15.1 PCA (Principal Component Analysis)

```
Problem: 100 features → hard to visualise, slow, curse of dimensionality

PCA: find the directions of MAXIMUM VARIANCE in the data and project onto them

Steps:
1. Centre data (subtract mean)
2. Compute covariance matrix
3. Find eigenvectors (principal components) and eigenvalues
4. Sort by eigenvalue (largest = most variance explained)
5. Project data onto top-k eigenvectors → reduced dimensions
```

```python
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Reduce 50 features to 2 for visualisation
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X)  # shape: (n_samples, 2)

# How much variance does each component explain?
print(f"Variance explained: {pca.explained_variance_ratio_}")
# E.g., [0.72, 0.15] = PC1 explains 72%, PC2 explains 15% = 87% total

# Choose n_components to retain 95% variance
pca_95 = PCA(n_components=0.95)
X_95 = pca_95.fit_transform(X)
print(f"Components for 95% variance: {pca_95.n_components_}")
```

### 15.2 t-SNE and UMAP (Visualisation Only)

```python
from sklearn.manifold import TSNE

# t-SNE: non-linear dimensionality reduction for VISUALISATION
# Always reduce to 2 or 3 dimensions
# Preserves local structure (nearby points stay nearby)
# WARNING: do not use for feature engineering or clustering!
# Distances and cluster sizes in t-SNE plots are NOT meaningful

tsne = TSNE(n_components=2, perplexity=30, random_state=42)
X_tsne = tsne.fit_transform(X)

# UMAP: faster alternative to t-SNE, also preserves global structure better
# pip install umap-learn
import umap
reducer = umap.UMAP(n_components=2, random_state=42)
X_umap = reducer.fit_transform(X)
```

---

<a name="part-16"></a>
## Part 16 — Feature Engineering

### 16.1 Feature Scaling

```python
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# StandardScaler (Z-score normalisation): mean=0, std=1
# Use for: linear models, neural networks, SVM, KNN, PCA
# Formula: x' = (x - mean) / std
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # use TRAIN statistics!

# MinMaxScaler: scale to [0, 1]
# Use for: neural networks, algorithms sensitive to scale
# Formula: x' = (x - min) / (max - min)
minmax = MinMaxScaler()
X_minmax = minmax.fit_transform(X_train)

# CRITICAL: fit scaler on TRAIN data only, then transform both train and test
# Fitting on test data = data leakage!
```

### 16.2 Encoding Categorical Variables

```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Label Encoding: map categories to integers
# "red"→0, "green"→1, "blue"→2
# WARNING: implies ordering (2 > 1 > 0) — only for ordinal features or tree models
le = LabelEncoder()
encoded = le.fit_transform(["red", "green", "blue", "red"])

# One-Hot Encoding: binary column per category
# "red"  → [1, 0, 0]
# "green"→ [0, 1, 0]
# "blue" → [0, 0, 1]
# Use for: linear models, neural networks (no false ordinal relationship)
df = pd.get_dummies(df, columns=['color'], drop_first=True)
# drop_first=True avoids multicollinearity (dummy variable trap)
```

### 16.3 Handling Missing Values

```python
# Option 1: Drop rows (if few missing)
df.dropna(inplace=True)

# Option 2: Impute with mean/median/mode
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')  # 'mean', 'median', 'most_frequent'
X_imputed = imputer.fit_transform(X)

# Option 3: Add "is_missing" indicator feature + impute
df['feature_is_missing'] = df['feature'].isna().astype(int)
df['feature'] = df['feature'].fillna(df['feature'].median())

# Note: tree-based models (XGBoost, LightGBM) handle missing values natively!
```

### 16.4 Feature Selection

```python
# Method 1: Correlation analysis — remove highly correlated features
corr_matrix = df.corr().abs()
upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [col for col in upper_triangle.columns if any(upper_triangle[col] > 0.95)]

# Method 2: Feature importance from tree models
model = RandomForestClassifier(n_estimators=100).fit(X, y)
importances = pd.Series(model.feature_importances_, index=feature_names)
importances.nlargest(20).plot(kind='barh')

# Method 3: Recursive Feature Elimination (RFE)
from sklearn.feature_selection import RFE
selector = RFE(estimator=LogisticRegression(), n_features_to_select=10)
selector.fit(X, y)
selected = feature_names[selector.support_]
```

---

<a name="part-17"></a>
## Part 17 — Probability & Statistics for Interviews

### 17.1 Bayes' Theorem (Used Everywhere)

$$P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}$$

```
Example interview question:
  "A disease affects 1% of the population. A test has 99% sensitivity (TPR)
   and 95% specificity (1-FPR). You test positive. What's the probability
   you actually have the disease?"

  P(disease) = 0.01 (prior — base rate)
  P(positive | disease) = 0.99 (sensitivity)
  P(positive | no disease) = 0.05 (1 - specificity = FPR)
  
  P(positive) = P(pos|disease)×P(disease) + P(pos|no disease)×P(no disease)
              = 0.99 × 0.01 + 0.05 × 0.99
              = 0.0099 + 0.0495
              = 0.0594
  
  P(disease | positive) = (0.99 × 0.01) / 0.0594 = 0.167 = 16.7%!
  
  Even with a positive test from a 99% accurate test,
  there's only a 16.7% chance you have the disease!
  → This is why BASE RATES matter (called the "base rate fallacy")
```

### 17.2 Distributions You Should Know

```
Bernoulli: Single coin flip. P(X=1) = p, P(X=0) = 1-p
Binomial: Number of successes in n independent Bernoulli trials. E[X]=np, Var=np(1-p)
Poisson: Number of events in a fixed interval. E[X]=λ, Var=λ
  Example: "Servers receive 10 requests/second. What's P(>15 in a second)?"
Normal (Gaussian): Bell curve. Described by mean μ and std σ.
  68% within ±1σ, 95% within ±2σ, 99.7% within ±3σ
Uniform: Equal probability everywhere in [a, b]
Exponential: Time between events in a Poisson process. E[X]=1/λ
```

### 17.3 Central Limit Theorem

```
No matter what the original distribution looks like, 
the MEAN of many independent samples from it approaches a NORMAL distribution.

Why it matters for ML:
  - Justifies why we can use normal-distribution-based tests for means
  - Explains why gradient averaging over mini-batches is stable
  - Why confidence intervals work

Formally: X̄ ~ Normal(μ, σ²/n) as n → ∞
The larger the sample size n, the tighter the distribution of the mean
```

### 17.4 Conditional Probability & Independence

```
P(A and B) = P(A|B) × P(B) = P(B|A) × P(A)

Independent events: P(A and B) = P(A) × P(B)
  → knowing A tells you nothing about B

Example: "Are user clicks on two different products independent?"
  → If P(click A | click B) = P(click A): yes, independent
  → If ≠: correlated (collaborative filtering exploits this!)
```

---

<a name="part-18"></a>
## Part 18 — A/B Testing & Hypothesis Testing

### 18.1 A/B Testing for ML Models

```
Goal: determine if Model B is better than Model A in production

Setup:
  Control (A): existing model gets 50% of traffic
  Treatment (B): new model gets 50% of traffic
  Metric: conversion rate, CTR, revenue per user, latency
  Duration: 1-4 weeks (depends on traffic and effect size)

Step 1: State hypotheses
  H₀ (null): Model B is NOT better than A (difference = 0)
  H₁ (alternative): Model B IS better than A
  
Step 2: Choose significance level (α = 0.05 typically)
  α = probability of rejecting H₀ when it's actually true (false positive)
  "5% chance of saying B is better when it's not"

Step 3: Run experiment, collect data

Step 4: Compute p-value
  p-value = probability of seeing the observed difference (or more) IF H₀ is true
  If p-value < α → reject H₀ → "B is significantly better than A"
  If p-value ≥ α → cannot reject H₀ → "not enough evidence"
```

### 18.2 Key Concepts

```
Type I Error (False Positive): saying B is better when it's not. Rate = α = 0.05
Type II Error (False Negative): saying B is NOT better when it IS. Rate = β
Power = 1 - β: probability of detecting a real improvement (target: 80%)

Sample size formula (for proportion test):
  n = (Z_α/2 + Z_β)² × (p₁(1-p₁) + p₂(1-p₂)) / (p₁ - p₂)²
  
  Higher power → need more samples
  Smaller effect size → need more samples
  
Common pitfall: running too short → underpowered test → can't detect real improvements
Common pitfall: peeking and stopping early → inflates false positive rate
```

### 18.3 Multiple Testing Correction

```
If you test 20 metrics simultaneously, even with α=0.05,
you expect 1 false positive just by chance!

Bonferroni correction: α_corrected = α / number_of_tests
  20 tests → α_corrected = 0.05/20 = 0.0025

Better: Benjamini-Hochberg (controls false discovery rate instead of FWER)
  Less conservative, more practical for multiple comparisons
```

---

<a name="part-19"></a>
## Part 19 — Interview Q&A: 50 Questions

### Fundamentals

**Q1: Explain the bias-variance tradeoff.**
> Total error = Bias² + Variance + Irreducible noise. Bias is systematic error from oversimplified assumptions (underfitting). Variance is sensitivity to fluctuations in training data (overfitting). Reducing one typically increases the other. The goal is the optimal balance — found through cross-validation and regularisation.

**Q2: What is regularisation and why do we use it?**
> Regularisation adds a penalty to the loss function that discourages model complexity, preventing overfitting. L1 (Lasso) penalises |w| and drives weights to zero for sparse models. L2 (Ridge) penalises w² and shrinks all weights evenly. ElasticNet combines both. In neural networks, dropout, early stopping, and weight decay serve as regularisers.

**Q3: What is gradient descent? Name three variants.**
> An optimisation algorithm that iteratively updates model parameters by moving in the direction of steepest loss decrease (negative gradient). Variants: Batch GD (entire dataset per step — stable but slow), SGD (one sample per step — fast but noisy), Mini-batch GD (B samples per step — practical standard, typically B=32-256). Adam adds adaptive learning rates per parameter and momentum.

**Q4: Why do we use cross-entropy loss for classification instead of MSE?**
> With sigmoid/softmax output, MSE creates a non-convex loss surface with many poor local minima, causing slow convergence. Cross-entropy produces a convex loss surface (for linear models), giving stronger gradients for wrong predictions (gradient proportional to error, not clipped by sigmoid saturation). Cross-entropy also has a probabilistic interpretation as negative log-likelihood.

**Q5: What is the difference between L1 and L2 regularisation?**
> L1 (Lasso): penalty $= \lambda\sum\lvert w_j \rvert$. Drives some weights exactly to zero → automatic feature selection. Creates sparse models. Gradient has constant magnitude → pulls toward zero equally regardless of weight size. L2 (Ridge): penalty $= \lambda\sum w_j^2$. Shrinks all weights toward zero but never exactly to zero. Gradient is proportional to weight → large weights are penalised more. ElasticNet combines both.

**Q6: When would you choose precision over recall?**
> When false positives are costly. Example: spam filter — marking a real email as spam is worse than letting some spam through. Email from your boss in spam folder → lost deal. Conversely, prioritise recall when false negatives are costly: cancer screening — missing a cancer case is worse than a false alarm.

**Q7: What is overfitting? How do you detect and prevent it?**
> Overfitting: model memorises training data including noise, performing well on training data but poorly on unseen data. Detected by: training error << test error, or high variance in cross-validation scores. Prevented by: more training data, regularisation (L1/L2/dropout), early stopping, simpler model, data augmentation, cross-validation for model selection.

**Q8: Explain the curse of dimensionality.**
> As the number of features increases: (1) distances between points converge — "nearest neighbour" becomes meaningless, (2) the volume of feature space grows exponentially — data becomes sparse, (3) more features = more parameters = higher risk of overfitting. Solutions: PCA, feature selection, regularisation, domain knowledge to pick relevant features.

### Models

**Q9: How does a decision tree decide which feature to split on?**
> It evaluates every (feature, threshold) combination and picks the one that maximally reduces impurity in the child nodes. Gini impurity measures 1 - Σpₖ², entropy measures -Σpₖlog₂(pₖ). The split with the highest information gain (parent impurity - weighted children impurity) is chosen. This is a greedy, locally-optimal algorithm.

**Q10: Why is Random Forest better than a single decision tree?**
> A single tree has high variance — small data changes produce a completely different tree. Random Forest reduces variance through two sources of randomness: (1) bootstrap sampling — each tree sees a random 63% of training data, (2) feature subsampling — each split considers only √p features. These decorrelate the trees so their errors cancel when averaged. Bias stays low because each tree is still complex.

**Q11: Explain gradient boosting in simple terms.**
> Build trees sequentially. Tree 1 fits the data. Tree 2 fits the ERRORS of Tree 1. Tree 3 fits the remaining errors. Each tree corrects mistakes of the ensemble so far. The final prediction is the sum of all trees' predictions. This reduces bias (can learn very complex patterns), but needs regularisation (learning rate, early stopping, max depth) to control variance.

**Q12: When would you use XGBoost vs a neural network?**
> XGBoost for structured/tabular data (customer churn, fraud detection, pricing) — especially when features are well-engineered, dataset is small-to-medium (<1M rows), interpretability matters (feature importance), or training speed matters. Neural networks for unstructured data (images, text, audio, video) — they learn features automatically from raw data. For tabular data, XGBoost often beats deep learning.

**Q13: What is SVM? When does it fail?**
> SVM finds the hyperplane that maximally separates classes. The margin (distance to nearest points — support vectors) is maximised. The kernel trick allows non-linear separation by implicitly mapping to higher dimensions. Fails when: dataset is large (O(n²-n³) training), highly noisy data (margin collapses), or when probability calibration is needed.

**Q14: What is K-Nearest Neighbours and when is it appropriate?**
> KNN: store all training data; for a new point, find K closest training points and majority-vote. No training needed, but O(n) at inference. Good for: small datasets, simple baselines, anomaly detection. Bad for: high dimensions (curse of dimensionality), large datasets (slow inference), when features have different scales (must normalise).

**Q15: Compare Naive Bayes to Logistic Regression for text classification.**
> Naive Bayes: generative model, assumes feature independence, very fast training, works well with very few samples, poor calibrated probabilities. Logistic Regression: discriminative model, no independence assumption, learns feature interactions implicitly, better calibrated probabilities, needs more data. For large datasets with enough data, Logistic Regression usually wins. For very small datasets or as a quick baseline, Naive Bayes.

### Evaluation & Practice

**Q16: What metric would you use for a highly imbalanced dataset (99% negative, 1% positive)?**
> NOT accuracy (a "predict all negative" model gets 99%!). Use: F1-score (harmonic mean of precision and recall), AUC-PR (precision-recall area, focuses on the rare positive class), or MCC (Matthews Correlation Coefficient — balanced even for imbalanced data). For threshold selection, use the precision-recall curve and pick the threshold matching business needs.

**Q17: What is AUC-ROC and how do you interpret it?**
> AUC-ROC measures the area under the Receiver Operating Characteristic curve (TPR vs FPR at all thresholds). AUC=1.0: perfect classifier. AUC=0.5: random. Interpretation: the probability that a random positive example is scored higher than a random negative example. Useful for comparing classifiers across all thresholds. Limitation: can be misleading for highly imbalanced data.

**Q18: What's the difference between k-fold cross-validation and a train/test split?**
> Train/test split: single evaluation, high variance — results depend on which data ended up in the test set. K-fold CV: K evaluations on K different test folds, then average — much more reliable performance estimate with confidence interval. Use stratified K-fold for classification. Time-series needs TimeSeriesSplit to avoid data leakage.

**Q19: How do you handle class imbalance?**
> Multiple approaches: (1) Resampling — oversample minority (SMOTE) or undersample majority, (2) Class weights — `class_weight='balanced'` in sklearn, (3) Change threshold — lower threshold for the rare class, (4) Use appropriate metrics (F1, AUC-PR, not accuracy), (5) Algorithmic — cost-sensitive learning, focal loss. In practice, I usually start with class weights and appropriate metrics.

**Q20: How do you decide between precision and recall for a business problem?**
> Ask: "What's the cost of a false positive vs a false negative?" If FP is costly (innocent jailed, good email to spam) → optimise precision. If FN is costly (cancer missed, fraud undetected) → optimise recall. In many cases, F1 balances both. Ultimately, translate to business metrics: "a false negative costs us $X, a false positive costs us $Y" → optimise for minimum total cost.

### Deep/Applied Questions

**Q21: Why does Adam work better than SGD in many cases?**
> Adam combines momentum (exponential moving average of gradients — smooths noise) with adaptive learning rates per parameter (divides by running average of squared gradients). Sparse features get larger updates, frequent features get smaller. Built-in bias correction for early steps. Works well "out of the box" with default hyperparameters. Limitation: may converge to wider minima than SGD with tuned schedule.

**Q22: What is the vanishing gradient problem?**
> In deep networks, gradients are multiplied through layers during backpropagation. With sigmoid/tanh activations, gradients are always < 1, so the product shrinks exponentially in early layers — they stop learning. Fixes: ReLU activation (gradient = 1 for positive inputs), residual connections (gradient flows directly through skip connections), batch/layer normalisation (keeps activations in good range).

**Q23: What are residual (skip) connections and why do they help?**
> Instead of learning h(x) = F(x), learn h(x) = F(x) + x (add the input to the output). If F needs to be identity (do nothing), it just needs to output zero — much easier. Residual connections enable training very deep networks (100+ layers) by creating a direct gradient path that avoids vanishing. Used in ResNet, all transformers, and modern LLMs.

**Q24: Explain batch normalisation.**
> During training, normalise each layer's activations to have mean=0, std=1 across the batch, then apply learnable scale (γ) and shift (β). Benefits: (1) Enables higher learning rates (activations don't explode), (2) Regularisation effect (each batch has slightly different statistics → noise → reduces overfitting), (3) Reduces sensitivity to initialisation. Modern transformers use Layer Norm instead (normalises across features, not batch).

**Q25: What is the difference between batch normalisation and layer normalisation?**
> Batch Norm: normalise across the batch dimension (mean/std computed from all examples in the batch for each feature). Depends on batch size — small batches degrade performance. Layer Norm: normalise across the feature dimension (mean/std computed from all features of a single example). Independent of batch size. Transformers use Layer Norm because sequence lengths vary and batch sizes are small.

**Q26: What is transfer learning?**
> Use a model pre-trained on a large dataset (ImageNet, internet text) and adapt it to your specific task. Two strategies: (1) Feature extraction — freeze the pre-trained layers, train only a new classification head, (2) Fine-tuning — unfreeze some/all layers and train end-to-end with a small learning rate. Works because early layers learn general features (edges, syntax) that transfer across tasks.

**Q27: How does data leakage happen and how do you prevent it?**
> Data leakage: information from the test set influences the training process, giving overly optimistic performance. Common causes: (1) Feature scaling fit on full dataset instead of train only, (2) Time-series: using future data to predict past, (3) Duplicate examples in train and test, (4) Target leakage (feature derived from the label). Prevention: always split first, then preprocess. Use pipelines.

**Q28: Explain stratified sampling.**
> When splitting data for train/test, stratified sampling ensures each split has the same class distribution as the full dataset. Critical for imbalanced datasets — without stratification, a random split might put all rare examples in one fold. In sklearn: `train_test_split(X, y, stratify=y)` and `StratifiedKFold`.

**Q29: What is feature importance and how is it calculated?**
> For tree models: importance = total reduction in impurity (Gini) from splits on that feature, summed across all trees. Features that appear in many splits near the root are most important. Permutation importance (model-agnostic): shuffle a feature's values and measure how much accuracy drops — bigger drop = more important. SHAP values: game-theoretic attribution of each feature's contribution to each prediction.

**Q30: How do you handle multicollinearity?**
> When features are highly correlated (VIF > 5-10). Problems: unstable coefficients in linear models, inflated standard errors. Solutions: (1) Remove one of the correlated pair, (2) PCA — combine correlated features, (3) L1/L2 regularisation — naturally handles it, (4) Tree-based models — inherently resistant (just use one of the correlated features at splits).

### System/Production Questions

**Q31: How do you monitor ML models in production?**
> Track: (1) Model metrics — accuracy/F1/AUC on production predictions with delayed ground truth, (2) Data drift — statistical tests comparing training vs production feature distributions (PSI, KS test), (3) Prediction drift — distribution of predictions changing, (4) Latency and throughput, (5) Business metrics — revenue, CTR, conversion. Alert when any metric degrades beyond threshold.

**Q32: What is concept drift?**
> The relationship between features and target changes over time. Example: COVID changed consumer buying patterns — a pre-COVID demand model fails post-COVID. Types: sudden (event-driven), gradual (evolving preferences), periodic/seasonal. Detection: monitor model metrics on recent data windows. Mitigation: scheduled retraining, continuous learning, adaptive models.

**Q33: How would you design a model retraining pipeline?**
> Trigger: scheduled (weekly/monthly), or metric-based (when performance drops below threshold). Pipeline: (1) Validated fresh data ingestion, (2) Feature engineering (reproducible), (3) Train + hyperparameter tune on recent data, (4) Evaluate against champion model on holdout, (5) If improved: deploy via shadow mode → A/B test → promote, (6) Versioning: MLflow for model artifacts and metrics.

**Q34: How do you handle a model that works well offline but poorly in production?**
> Common causes: (1) Training-serving skew — different preprocessing in training vs serving, (2) Data leakage — test set was contaminated during training, (3) Distribution shift — production data differs from training data, (4) Feature freshness — features stale or delayed in production, (5) Latency constraints — model too slow, timeout causes partial results. Debug systematically: compare feature distributions, check preprocessing code, log actual production inputs.

**Q35: What's the difference between online learning and batch learning?**
> Batch learning: train on entire historical dataset, deploy, retrain periodically. Online learning: update model incrementally with each new data point (or small batch). Online is critical for: fast-changing distributions (ad CTR, stock prices), massive datasets that don't fit in memory, personalisation in real-time. Algorithms: SGD-based models (logistic regression), online random forest, Vowpal Wabbit.

### Probability & Statistics

**Q36: A test is 99% accurate. You test positive. What's the probability you're sick?**
> It depends on the base rate! If 1% of the population is sick: P(sick|positive) = (0.99×0.01) / (0.99×0.01 + 0.01×0.99) = 0.5 (50%). If 0.1%: P(sick|positive) ≈ 9%. This is Bayes' theorem in action. The key insight: for rare conditions, even highly accurate tests have high false positive rates relative to true positives.

**Q37: What is p-value? Is p < 0.05 enough?**
> The p-value is the probability of observing the test result (or more extreme) assuming H₀ is true. It is NOT the probability that H₀ is true. p < 0.05 means: "if there's no real effect, there's < 5% chance of seeing this result." Criticisms: (1) 0.05 is arbitrary, (2) doesn't measure effect SIZE, (3) with large n, tiny meaningless effects become "significant." Always report confidence intervals and effect sizes alongside p-values.

**Q38: What is selection bias?**
> The sample used for training is not representative of the production population. Examples: (1) Survivorship bias — model trained only on customers who stayed (missing churned customers), (2) Sampling bias — data collected only from one region/demographic, (3) Label bias — positive examples are easier to identify than negative. Fix: deliberate stratified sampling, careful analysis of data collection process, monitoring for distribution shift.

**Q39: Explain the law of large numbers.**
> As the number of independent, identically distributed trials increases, the sample mean converges to the expected value. Practical meaning: with enough data, your metrics become reliable. Why mini-batch gradients work: sampling noise decreases as batch size increases. Why A/B tests need enough traffic: small sample → unreliable results.

**Q40: What is Simpson's Paradox?**
> A trend present in several groups reverses when the groups are combined. Example: Drug A has higher recovery rate than B overall. But within each age group, Drug B has higher recovery rate. Caused by a confounding variable (age distribution differs between groups). Relevance to ML: feature importance or model performance can be misleading without proper disaggregation.

### Advanced/Senior Questions

**Q41: How would you approach building an ML model from scratch for a new business problem?**
> 1. Define the problem as ML (predict what? from what input? what metric matters?), 2. Collect and explore data (distributions, missing values, correlations), 3. Establish a baseline (simple heuristic or logistic regression), 4. Feature engineering (domain knowledge), 5. Train/evaluate multiple models (cross-validation, not just accuracy), 6. Error analysis on validation set (where does the model fail?), 7. Iterate on features/model/data, 8. Deploy with monitoring and A/B test.

**Q42: What would you do if your model has high training accuracy but low test accuracy?**
> This is overfitting (high variance). Systematic approach: (1) More training data (the best fix), (2) Regularisation — L1/L2, dropout, weight decay, (3) Simpler model — reduce layers, features, tree depth, (4) Early stopping, (5) Data augmentation, (6) Feature selection — remove noisy features, (7) Ensemble methods (bagging reduces variance). Diagnose with learning curves (plot train/test error vs training set size).

**Q43: What would you do if both training and test accuracy are low?**
> This is underfitting (high bias). Fixes: (1) More complex model — more layers, deeper trees, polynomial features, (2) More/better features — domain knowledge, (3) Less regularisation, (4) Train longer (more epochs), (5) Reduce learning rate (for neural nets — may be overshooting). If still low: the problem may not be solvable with the given features (need different data).

**Q44: How do you determine if a feature is useful?**
> Multiple methods: (1) Correlation with target (but captures linear only), (2) Mutual information (captures non-linear), (3) Train model with and without the feature — compare CV scores, (4) Permutation importance — shuffle the feature and measure accuracy drop, (5) SHAP values — contribution per prediction. Avoid removing features based only on correlation with target; some features are important in combination.

**Q45: What is the difference between generative and discriminative models?**
> Discriminative: model P(y|x) directly — the decision boundary. Examples: logistic regression, SVM, neural networks. Generative: model P(x|y)·P(y) — the full distribution of each class, then derive P(y|x) via Bayes. Examples: Naive Bayes, Gaussian Mixture Models, VAEs, GPT (generates text). Discriminative usually achieves better classification accuracy with enough data. Generative can generate new data and handle missing features.

**Q46: What are embeddings in ML?**
> Dense, learned vector representations of discrete entities (words, products, users). Instead of one-hot encoding (sparse, no similarity), embeddings capture semantic relationships in continuous space. "King" and "queen" have similar embeddings. Used in: Word2Vec, LLM token embeddings, recommendation systems (user/item embeddings), categorical feature encoding in deep learning.

**Q47: Explain the difference between parametric and non-parametric models.**
> Parametric: fixed number of parameters regardless of data size. Linear regression (d weights + bias), logistic regression, naive bayes. Assumptions about data distribution. Fast prediction. Non-parametric: number of effective parameters grows with data. KNN (stores all data), decision trees (nodes grow with data), kernel SVM. Fewer assumptions, more flexible, but slower/more memory at inference.

**Q48: What is the No Free Lunch Theorem?**
> No single algorithm is best for all problems. An algorithm that works well on one class of problems must work poorly on another class. Implication: always benchmark multiple algorithms on your specific data. There's no universal "best" model — only the best model for your particular problem. This is why ML involves experimentation and evaluation, not just picking the "latest" model.

**Q49: How do you handle a situation where you have very little labelled data?**
> (1) Transfer learning — use pre-trained model (ImageNet, BERT, LLM), (2) Semi-supervised learning — pseudo-label unlabelled data, (3) Data augmentation — create synthetic variations, (4) Active learning — strategically label most informative examples, (5) Few-shot learning — prompt LLMs with few examples, (6) Weak supervision — use heuristic labelling functions (Snorkel), (7) Self-supervised pre-training on unlabelled data.

**Q50: What are the key differences between classical ML and deep learning?**
> Classical ML: requires manual feature engineering, works well on structured/tabular data, interpretable (decision trees, linear models), efficient on small datasets. Deep Learning: learns features automatically from raw data, excels on unstructured data (images, text, audio), requires large datasets and GPUs, less interpretable but more powerful for complex patterns. For tabular data, XGBoost often beats deep learning. For language/vision, deep learning dominates.

---

<a name="part-20"></a>
## 📚 Further Resources

**Books:**
- **"Hands-On Machine Learning with Scikit-Learn, Keras, and TensorFlow" by Aurélien Géron** — The best practical ML book (covers everything in this guide with code)
- **"An Introduction to Statistical Learning (ISLR)" by James, Witten, Hastie, Tibshirani** — Free PDF, excellent theory with R examples. https://www.statlearning.com
- **"The Elements of Statistical Learning" by Hastie, Tibshirani, Friedman** — Advanced theory (grad-level). Free PDF.

**Courses:**
- **Andrew Ng — Machine Learning Specialization (Coursera)** — The classic, recently updated
- **StatQuest (YouTube, Josh Starmer)** — Brilliant visual explanations of every concept in this file
- **3Blue1Brown — Neural Networks (YouTube)** — Beautiful visualisation of gradient descent and backpropagation

**Practice:**
- Scikit-learn documentation: https://scikit-learn.org (examples for every algorithm)
- Kaggle Learn: https://www.kaggle.com/learn (free micro-courses with hands-on exercises)

> **This file covers the entire "ML Fundamentals" round for AI engineer interviews.** Combine this knowledge with the LLM-specific content in the other study guides, and you'll be able to handle any question they throw at you.
