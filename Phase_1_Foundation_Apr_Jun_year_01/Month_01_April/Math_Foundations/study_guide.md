# Math Foundations for AI/ML — Complete Study Guide

> **Excel Curriculum Coverage**: Linear Algebra, Calculus, Probability & Statistics
> **Interview Focus**: Matrix operations, gradient computation, probability distributions, Bayes' theorem
> **Day-to-Day**: Every neural network is linear algebra + calculus — understanding math means understanding models

---

## Table of Contents
1. [Linear Algebra](#1-linear-algebra)
2. [Calculus for ML](#2-calculus-for-ml)
3. [Probability & Statistics Deep Dive](#3-probability--statistics)
4. [Interview Questions (30 Q&As)](#4-interview-questions)
5. [Day-to-Day Work Applications](#5-day-to-day-work-applications)
6. [Resources](#6-resources)

---

## 1. Linear Algebra

### Vectors

```python
import numpy as np

# Vector operations
a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# Dot product: a · b = |a||b|cos(θ)
dot = np.dot(a, b)    # 1*4 + 2*5 + 3*6 = 32

# Norms
l1_norm = np.linalg.norm(a, ord=1)    # |1|+|2|+|3| = 6 (Manhattan)
l2_norm = np.linalg.norm(a, ord=2)    # √(1²+2²+3²) = 3.74 (Euclidean)
linf_norm = np.linalg.norm(a, ord=np.inf)  # max(|1|,|2|,|3|) = 3

# Cosine similarity (used EVERYWHERE in embeddings)
cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
# This is how RAG retrieval works — cosine similarity between query and document embeddings
```

### Why Cosine Similarity Matters for AI Engineers

$$\text{cosine\_sim}(a, b) = \frac{a \cdot b}{\|a\| \|b\|} = \frac{\sum a_i b_i}{\sqrt{\sum a_i^2} \sqrt{\sum b_i^2}}$$

- Range: [-1, 1] (1 = identical direction, 0 = orthogonal, -1 = opposite)
- Used in: RAG retrieval, semantic search, recommendation systems, embedding comparison
- Normalized dot product — ignores magnitude, captures direction/meaning

### Matrices

```python
# Matrix as transformation
A = np.array([[2, 0], [0, 3]])  # Scaling matrix
v = np.array([1, 1])
result = A @ v  # [2, 3] — scales x by 2, y by 3

# Matrix multiplication: (m×k) @ (k×n) = (m×n)
W = np.random.randn(768, 3072)   # Weight matrix (like in Transformer FFN)
x = np.random.randn(32, 768)     # Input batch (32 samples, 768 features)
output = x @ W                    # (32, 3072) — linear transformation

# Transpose
A = np.array([[1, 2, 3], [4, 5, 6]])  # (2, 3)
A_T = A.T                              # (3, 2)

# Identity matrix
I = np.eye(4)  # 4×4 identity — A @ I = A

# Matrix inverse
A = np.array([[4, 7], [2, 6]])
A_inv = np.linalg.inv(A)
print(A @ A_inv)  # ≈ Identity matrix
```

### Eigenvalues & Eigenvectors

$$A v = \lambda v$$

Where $v$ is an eigenvector and $\lambda$ is the corresponding eigenvalue.

**Intuition**: An eigenvector is a direction that doesn't change under the transformation — it only gets scaled by $\lambda$.

```python
A = np.array([[4, 1], [2, 3]])
eigenvalues, eigenvectors = np.linalg.eig(A)
# eigenvalues: [5, 2]
# The matrix stretches by 5x in one direction and 2x in another

# Verify: A @ v = λ * v
for i in range(len(eigenvalues)):
    v = eigenvectors[:, i]
    lam = eigenvalues[i]
    print(f"A @ v = {A @ v}, λ*v = {lam * v}")  # Should be equal
```

**Where eigenvalues appear in ML**:
- **PCA**: Principal components are eigenvectors of the covariance matrix
- **Google's PageRank**: Dominant eigenvector of the link matrix
- **Spectral clustering**: Eigenvectors of the graph Laplacian

### Singular Value Decomposition (SVD)

$$A = U \Sigma V^T$$

- $U$: Left singular vectors (m×m orthogonal)
- $\Sigma$: Singular values (m×n diagonal, sorted descending)
- $V^T$: Right singular vectors (n×n orthogonal)

```python
A = np.random.randn(100, 50)
U, sigma, Vt = np.linalg.svd(A, full_matrices=False)

# Low-rank approximation (dimensionality reduction)
k = 10  # Keep top 10 components
A_approx = U[:, :k] @ np.diag(sigma[:k]) @ Vt[:k, :]
# A_approx ≈ A with much less data (compression)

# How much information is retained?
energy = np.sum(sigma[:k]**2) / np.sum(sigma**2)
print(f"Top {k} components retain {energy:.1%} of variance")
```

**SVD in ML/AI**:
- **LoRA** (Low-Rank Adaptation): W + ΔW where ΔW = BA is a low-rank update — inspired by SVD
- **Dimensionality reduction**: Alternative to PCA
- **Recommendation systems**: Matrix factorization for collaborative filtering
- **Data compression**: Reduce embedding dimensions

### PCA (Principal Component Analysis)

```python
from sklearn.decomposition import PCA

# Reduce 768-dimensional embeddings to 2D for visualization
embeddings = np.random.randn(1000, 768)  # 1000 embeddings, 768 dims

pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

print(f"Explained variance: {pca.explained_variance_ratio_.sum():.1%}")

# Choosing number of components
pca_full = PCA()
pca_full.fit(embeddings)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
n_components_95 = np.argmax(cumulative_variance >= 0.95) + 1
print(f"Need {n_components_95} components for 95% variance")
```

---

## 2. Calculus for ML

### Derivatives — The Foundation of Training

The derivative measures how a function changes as its input changes:
$$f'(x) = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$$

**Key derivatives for ML**:

| Function | Derivative | Used In |
|----------|-----------|---------|
| $x^n$ | $nx^{n-1}$ | Polynomial features |
| $e^x$ | $e^x$ | Softmax, losses |
| $\ln(x)$ | $1/x$ | Log-loss, cross-entropy |
| $\sigma(x) = \frac{1}{1+e^{-x}}$ | $\sigma(x)(1-\sigma(x))$ | Sigmoid activation |
| $\tanh(x)$ | $1-\tanh^2(x)$ | Tanh activation |
| $\text{ReLU}(x) = \max(0,x)$ | $\begin{cases} 0 & x < 0 \\ 1 & x > 0 \end{cases}$ | ReLU activation |

### Partial Derivatives & Gradients

For a function of multiple variables, the gradient is the vector of all partial derivatives:

$$\nabla f(x, y) = \left(\frac{\partial f}{\partial x}, \frac{\partial f}{\partial y}\right)$$

```python
# Example: MSE Loss
# L = (1/n) Σ (y_i - (wx_i + b))²
# ∂L/∂w = -(2/n) Σ x_i(y_i - (wx_i + b))  ← how loss changes with weight
# ∂L/∂b = -(2/n) Σ (y_i - (wx_i + b))      ← how loss changes with bias

def compute_gradients(x, y, w, b):
    """Manual gradient computation for linear regression."""
    n = len(x)
    predictions = w * x + b
    errors = y - predictions
    
    dw = -(2/n) * np.dot(x, errors)   # Gradient w.r.t. weight
    db = -(2/n) * np.sum(errors)       # Gradient w.r.t. bias
    
    return dw, db

def gradient_descent(x, y, lr=0.01, epochs=1000):
    w, b = 0.0, 0.0
    for _ in range(epochs):
        dw, db = compute_gradients(x, y, w, b)
        w -= lr * dw  # Update: move opposite to gradient
        b -= lr * db
    return w, b
```

### The Chain Rule — Backbone of Backpropagation

$$\frac{df}{dx} = \frac{df}{dg} \cdot \frac{dg}{dx}$$

```
Example: f(x) = (2x + 3)²

Let g(x) = 2x + 3, then f(g) = g²

df/dx = df/dg · dg/dx = 2g · 2 = 4(2x + 3)
```

**In neural networks**: Loss → output layer → hidden layers → input
Each layer's gradient = local gradient × upstream gradient

```python
# Chain rule in a 3-layer network:
# x → [W1] → h1 → [W2] → h2 → [W3] → y → Loss

# ∂Loss/∂W3 = ∂Loss/∂y · ∂y/∂W3
# ∂Loss/∂W2 = ∂Loss/∂y · ∂y/∂h2 · ∂h2/∂W2
# ∂Loss/∂W1 = ∂Loss/∂y · ∂y/∂h2 · ∂h2/∂h1 · ∂h1/∂W1

# This is exactly what PyTorch autograd computes automatically!
```

### The Jacobian Matrix

For vector-valued functions, the Jacobian generalizes the derivative:

$$J = \begin{bmatrix} \frac{\partial f_1}{\partial x_1} & \cdots & \frac{\partial f_1}{\partial x_n} \\ \vdots & \ddots & \vdots \\ \frac{\partial f_m}{\partial x_1} & \cdots & \frac{\partial f_m}{\partial x_n} \end{bmatrix}$$

Used in: understanding how attention outputs change w.r.t. inputs, advanced optimization (Newton's method).

### Softmax Gradient

$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_j e^{z_j}}$$

$$\frac{\partial \text{softmax}(z_i)}{\partial z_j} = \text{softmax}(z_i)(\delta_{ij} - \text{softmax}(z_j))$$

```python
def softmax(z):
    exp_z = np.exp(z - np.max(z))  # Subtract max for numerical stability
    return exp_z / exp_z.sum()

z = np.array([2.0, 1.0, 0.1])
print(softmax(z))  # [0.659, 0.242, 0.099]
```

---

## 3. Probability & Statistics

### Probability Distributions

```python
import numpy as np
from scipy import stats

# --- Normal (Gaussian) Distribution ---
# P(x) = (1/√(2πσ²)) · exp(-(x-μ)²/(2σ²))
mu, sigma = 0, 1
samples = np.random.normal(mu, sigma, 10000)
print(f"Mean: {np.mean(samples):.3f}, Std: {np.std(samples):.3f}")

# 68-95-99.7 rule:
# 68% within 1σ, 95% within 2σ, 99.7% within 3σ

# --- Bernoulli Distribution ---
# Single trial with probability p
p = 0.7
samples = np.random.binomial(1, p, 10000)

# --- Binomial Distribution ---
# n trials, each with probability p
n, p = 10, 0.5
samples = np.random.binomial(n, p, 10000)

# --- Poisson Distribution ---
# Events per time interval (rate λ)
lam = 5  # Average 5 events per interval
samples = np.random.poisson(lam, 10000)

# --- Uniform Distribution ---
samples = np.random.uniform(0, 1, 10000)
```

### Bayes' Theorem

$$P(A \mid B) = \frac{P(B \mid A) \cdot P(A)}{P(B)}$$

**Intuition**: Update your belief about A after observing evidence B.

```python
# Example: Spam filter
# P(Spam|"buy now") = P("buy now"|Spam) × P(Spam) / P("buy now")

p_spam = 0.3                      # Prior: 30% of emails are spam
p_buy_given_spam = 0.8            # "buy now" appears in 80% of spam
p_buy_given_not_spam = 0.1        # "buy now" appears in 10% of non-spam
p_buy = p_buy_given_spam * p_spam + p_buy_given_not_spam * (1 - p_spam)

p_spam_given_buy = (p_buy_given_spam * p_spam) / p_buy
print(f"P(Spam|'buy now') = {p_spam_given_buy:.3f}")  # 0.774
```

**Bayes in ML/AI**:
- Naive Bayes classifier
- Bayesian optimization (hyperparameter tuning)
- Bayesian neural networks
- Prior/posterior in generative models (VAEs)

### Conditional Probability & Independence

```python
# P(A|B) = P(A ∩ B) / P(B)
# Independent: P(A|B) = P(A), meaning P(A ∩ B) = P(A) × P(B)

# Example: Rolling a die
# P(Even|>3) = P(Even AND >3) / P(>3)
# P(Even AND >3) = P({4, 6}) = 2/6
# P(>3) = P({4, 5, 6}) = 3/6
# P(Even|>3) = (2/6) / (3/6) = 2/3
```

### Expected Value & Variance

$$E[X] = \sum x_i P(x_i) \quad \text{(discrete)}$$
$$\text{Var}(X) = E[(X - \mu)^2] = E[X^2] - (E[X])^2$$

```python
# For a fair die
outcomes = [1, 2, 3, 4, 5, 6]
probabilities = [1/6] * 6

expected_value = sum(x * p for x, p in zip(outcomes, probabilities))  # 3.5
variance = sum((x - expected_value)**2 * p for x, p in zip(outcomes, probabilities))  # 2.917
```

### Hypothesis Testing

```python
from scipy import stats

# A/B Test: Did our new prompt improve conversion?
# H0: No difference (conversion rates are equal)
# H1: New prompt is better

control_conversions = 120
control_total = 1000
treatment_conversions = 145
treatment_total = 1000

# Two-proportion z-test
z_stat, p_value = stats.proportions_ztest(
    [treatment_conversions, control_conversions],
    [treatment_total, control_total],
    alternative='larger'
)

alpha = 0.05
if p_value < alpha:
    print(f"Statistically significant (p={p_value:.4f}). New prompt is better.")
else:
    print(f"Not significant (p={p_value:.4f}). Keep current prompt.")
```

### Key Statistical Concepts for AI

| Concept | What It Means | Where Used |
|---------|--------------|------------|
| **Mean, Median, Mode** | Central tendency | Data analysis, feature engineering |
| **Standard Deviation** | Spread of data | Normalization, anomaly detection |
| **Correlation** | Linear relationship between variables | Feature selection |
| **Confidence Interval** | Range likely containing true parameter | A/B testing, model evaluation |
| **p-value** | Probability of data given H0 is true | Significance testing |
| **Type I Error** | False positive (reject true H0) | α in testing |
| **Type II Error** | False negative (fail to reject false H0) | Power analysis |
| **Central Limit Theorem** | Sample means → Normal for large n | Justifies many statistical tests |
| **Maximum Likelihood Estimation** | Find parameters that maximize data probability | Training ML models |
| **Cross-Entropy** | Measures difference between distributions | Loss function for classification |

### Information Theory

$$H(X) = -\sum P(x) \log_2 P(x) \quad \text{(entropy)}$$
$$H(P, Q) = -\sum P(x) \log Q(x) \quad \text{(cross-entropy)}$$
$$D_{KL}(P \| Q) = \sum P(x) \log \frac{P(x)}{Q(x)} \quad \text{(KL divergence)}$$

```python
def entropy(probs):
    return -np.sum(probs * np.log2(probs + 1e-10))

def cross_entropy(p, q):
    return -np.sum(p * np.log(q + 1e-10))

def kl_divergence(p, q):
    return np.sum(p * np.log(p / (q + 1e-10) + 1e-10))

# High entropy = high uncertainty (uniform distribution)
# Low entropy = low uncertainty (peaked distribution)
print(entropy([0.25, 0.25, 0.25, 0.25]))  # 2.0 (maximum uncertainty)
print(entropy([0.97, 0.01, 0.01, 0.01]))  # 0.24 (very certain)
```

**Why information theory matters for AI**:
- **Cross-entropy loss**: The standard loss function for classification and language modeling
- **KL divergence**: Used in VAEs, RLHF (KL penalty), knowledge distillation
- **Entropy**: Measures model uncertainty, used in active learning
- **Mutual information**: Feature selection, neural network analysis

---

## 4. Interview Questions

### Q1: Why is linear algebra important for ML?
**A**: Neural networks are sequences of matrix multiplications with non-linear activations. Weights are matrices, inputs are vectors. Training requires computing gradients through chains of matrix operations. Understanding linear algebra means understanding: how data is transformed layer-by-layer, why certain architectures work, and how to debug dimensional issues.

### Q2: What is cosine similarity and why is it used for embeddings?
**A**: Cosine similarity measures the angle between two vectors, normalized to [-1, 1]. cos(a,b) = (a·b)/(‖a‖‖b‖). Used for embeddings because: (1) Scale-invariant — embedding magnitude doesn't affect similarity. (2) Efficient — can be computed with normalized dot products. (3) Meaningful — similar meaning = similar direction in embedding space. Foundation of RAG retrieval.

### Q3: Explain eigenvalue decomposition intuitively.
**A**: A matrix can be understood as a transformation. Eigenvalues tell you HOW MUCH it stretches in each principal direction. Eigenvectors tell you WHICH directions are stretched. Decomposing A = QΛQ⁻¹ means: rotate to eigenvector basis (Q⁻¹), scale by eigenvalues (Λ), rotate back (Q). In PCA: eigenvectors of the covariance matrix = principal components.

### Q4: What is SVD and how does it relate to LoRA?
**A**: SVD decomposes any matrix A = UΣV^T. For a rank-r approximation, keep top r singular values. LoRA insight: weight updates during fine-tuning are low-rank. Instead of updating W (d×d = millions of parameters), learn ΔW = BA where B is (d×r) and A is (r×d) with r << d. This is equivalent to a low-rank factorization, saving 99%+ of parameters.

### Q5: Explain the chain rule and why it's essential for deep learning.
**A**: Chain rule: d/dx[f(g(x))] = f'(g(x)) · g'(x). In neural networks, loss depends on output which depends on each layer's weights through a chain of functions. Backpropagation IS the chain rule applied from output to input layer. Each layer computes: local gradient × upstream gradient. This is why vanishing/exploding gradients happen — many multiplications.

### Q6: What is gradient descent intuitively?
**A**: You're blindfolded on a hilly landscape trying to find the lowest point. At each step: (1) Feel the slope (compute gradient). (2) Take a step downhill (opposite to gradient). (3) Repeat. Learning rate = step size. Too large: overshoot. Too small: too slow. The gradient points in the direction of steepest ascent; we go opposite for descent.

### Q7: What's the difference between L1 and L2 norms?
**A**: L1 (Manhattan): sum of absolute values ‖x‖₁ = Σ|xᵢ|. Creates sparse solutions (many zeros). L2 (Euclidean): square root of sum of squares ‖x‖₂ = √Σxᵢ². Shrinks all values uniformly (no zeros). In regularization: L1 → feature selection (Lasso). L2 → smooth weights (Ridge). L2 is the default in deep learning (weight decay).

### Q8: Explain Bayes' theorem with a practical example.
**A**: P(A|B) = P(B|A)·P(A)/P(B). Example: 1% of emails are phishing. A detection system flags 90% of phishing (true positive) and 5% of legitimate emails (false positive). P(phishing|flagged) = (0.9×0.01)/(0.9×0.01 + 0.05×0.99) = 0.154. Only 15.4% of flagged emails are actually phishing! This is why precision matters — base rate (prior) dominates.

### Q9: What is the softmax function and why is it used?
**A**: $\text{softmax}(z_i) = \exp(z_i) \;/\; \sum_j \exp(z_j)$. Converts raw scores (logits) to a probability distribution (positive, sums to 1). Used as the final layer for multi-class classification and in attention mechanisms. The exponential amplifies differences — larger logits get much more probability. Temperature scaling ($z/T$) controls how "sharp" the distribution is.

### Q10: Explain the relationship between maximum likelihood and cross-entropy.
**A**: Maximizing log-likelihood is equivalent to minimizing cross-entropy. For classification with true label y and predicted probability q: log-likelihood = log(q_y). Cross-entropy = -Σ yᵢ log(qᵢ). When labels are one-hot, cross-entropy equals -log(q_y) = negative log-likelihood. This is why cross-entropy is THE standard classification loss.

### Q11: What is the curse of dimensionality?
**A**: As dimensions increase, data becomes increasingly sparse. In high dimensions: (1) All points are roughly equidistant (distance metrics break). (2) Volume concentrates near the surface of a hypersphere. (3) Need exponentially more data to cover the space. Solutions: dimensionality reduction (PCA), feature selection, regularization. Embeddings combat this by learning compact representations.

### Q12: Explain PCA step by step.
**A**: (1) Center data (subtract mean). (2) Compute covariance matrix. (3) Find eigenvalues/eigenvectors. (4) Sort by eigenvalue (descending). (5) Keep top k eigenvectors. (6) Project data onto k-dimensional subspace. The first principal component captures the most variance, second captures the most remaining variance (orthogonal to first), etc.

### Q13: What is the difference between covariance and correlation?
**A**: Covariance: Cov(X,Y) = E[(X-μX)(Y-μY)]. Measures joint variability but depends on scale. Correlation: ρ = Cov(X,Y)/(σX·σY). Normalized to [-1,1]. Scale-independent. Correlation = 0 means no linear relationship (but non-linear relationship may exist). Both measure linear dependencies.

### Q14: Explain cross-entropy loss mathematically.
**A**: H(p,q) = -Σ p(x)·log(q(x)). For classification: p is the true distribution (one-hot), q is the model's predicted probabilities. Simplifies to: L = -log(q_y) where y is the true class. Minimizing cross-entropy = maximizing the probability assigned to the correct class. For language modeling: average cross-entropy over all tokens.

### Q15: What is KL divergence and where is it used?
**A**: KL(P‖Q) = Σ P(x)·log(P(x)/Q(x)). Measures how different Q is from P. Not symmetric (KL(P‖Q) ≠ KL(Q‖P)). Always ≥ 0, = 0 only when P=Q. Used in: VAE loss (regularize latent space toward N(0,I)), RLHF (keep fine-tuned model close to base), knowledge distillation (student approximates teacher distribution).

### Q16: What is the Central Limit Theorem and why does it matter?
**A**: For large enough samples (n > 30), the distribution of sample means approaches a Normal distribution, regardless of the underlying distribution. Mean of sample means = population mean. Std of sample means = σ/√n. Matters because it justifies: confidence intervals, hypothesis testing, and many statistical inference methods. Enables A/B testing.

### Q17: Explain the concept of entropy in information theory.
**A**: Entropy H(X) = -Σ P(x)·log₂(P(x)). Measures uncertainty/information content. Maximum when all outcomes equally likely (uniform distribution). Minimum (0) when outcome is certain. For a coin: fair coin H=1 bit, biased (0.99/0.01) H≈0.08 bits. In ML: measures how uncertain a model is about its predictions.

### Q18: How does matrix multiplication work in neural networks?
**A**: Input x (batch_size × features) multiplied by weight W (features × output_dim) gives output (batch_size × output_dim). Each output neuron computes a weighted sum of inputs plus bias: z = Wx + b. Deep networks chain these: z = W₃(σ(W₂(σ(W₁x + b₁)) + b₂)) + b₃. Shape compatibility is critical — this is where dim mismatch errors come from.

### Q19: What is the Jacobian and when do you need it?
**A**: The Jacobian is the matrix of all partial derivatives of a vector-valued function. For $f: \mathbb{R}^n \to \mathbb{R}^m$, $J$ is $m \times n$ where $J_{ij} = \partial f_i / \partial x_j$. In deep learning: autograd computes Jacobian-vector products efficiently (without forming the full Jacobian). Needed for: understanding how attention changes w.r.t. inputs, Newton's method optimization, and theoretical analysis.

### Q20: Explain p-values and confidence intervals.
**A**: p-value: Probability of observing data as extreme as ours, assuming H0 is true. p < 0.05 → reject H0 (significant). Common misconception: p-value is NOT the probability H0 is false. Confidence interval: Range estimated to contain the true parameter with X% probability. 95% CI means: if we repeated the experiment 100 times, ~95 CIs would contain the true value.

### Q21: What is Maximum Likelihood Estimation?
**A**: Find parameter values that maximize the probability of observing the data. θ_MLE = argmax P(data|θ). In practice: maximize log-likelihood (easier, equivalent). For linear regression with Gaussian noise: MLE = least squares. For logistic regression: MLE → cross-entropy loss. Neural network training is essentially MLE/MAP estimation.

### Q22: What is a positive definite matrix and why does it matter?
**A**: A matrix M is positive definite if x^T M x > 0 for all non-zero x. Equivalently: all eigenvalues are positive. Matters because: covariance matrices are positive semi-definite, loss function Hessians should be positive definite at a minimum (convex), kernel matrices must be positive semi-definite. Ensures optimization has well-defined minima.

### Q23: Explain the difference between L1 and L2 regularization mathematically.
**A**: L1: $\mathcal{L}_{\text{reg}} = \mathcal{L} + \lambda\sum\lvert w_i \rvert$. Gradient is $\pm\lambda$ (constant push toward zero) → creates exact zeros → sparsity. L2: $\mathcal{L}_{\text{reg}} = \mathcal{L} + \lambda\sum w_i^2$. Gradient is $2\lambda w_i$ (proportional to weight) → shrinks proportionally → no exact zeros. L1 for feature selection (sparse models), L2 for general regularization (all features contribute but with smaller weights).

### Q24: What is the relationship between dot product and projection?
**A**: proj_b(a) = (a·b/‖b‖²)·b. The dot product a·b = ‖a‖‖b‖cos(θ) measures how much a aligns with b. In attention: Q·K^T computes how much each query aligns with each key — essentially projecting queries onto keys to find relevant information. Self-attention is a learned projection/alignment mechanism.

### Q25: How do you compute the gradient of the cross-entropy loss w.r.t. logits?
**A**: For CE loss with softmax: L = -log(softmax(z_y)) where z is logits and y is true class. The gradient has a beautifully simple form: ∂L/∂zᵢ = softmax(zᵢ) - yᵢ (where y is one-hot). This simplicity is why softmax + cross-entropy is the standard choice — clean gradients enable stable training.

### Q26: What is the Hessian matrix?
**A**: Matrix of second-order partial derivatives: $H_{ij} = \partial^2 f / \partial x_i \partial x_j$. Describes the curvature of the loss surface. Positive definite Hessian → local minimum. Used in: Newton's method (second-order optimization), understanding loss landscapes, and natural gradient methods. Too expensive to compute for large models ($n^2$ entries).

### Q27: Explain the difference between population and sample statistics.
**A**: Population: parameters of the entire distribution (μ, σ²). Sample: estimates from observed data (x̄, s²). Key difference: sample variance uses n-1 (Bessel's correction) instead of n to be unbiased. In ML: we always work with samples (training data) to estimate population properties. More data → better estimates (law of large numbers).

### Q28: What is Monte Carlo estimation?
**A**: Approximate expectations by averaging random samples: E[f(X)] ≈ (1/n)Σf(xᵢ) where xᵢ ~ P(X). Used extensively in: RL (estimating returns), VAEs (estimating ELBO), MCMC for Bayesian inference, and dropout (averaging over random masks). Quality improves with √n — need 4× samples for 2× precision.

### Q29: Why is numerical stability important in ML?
**A**: Floating point has limited precision. log(0) = -inf, exp(1000) = inf. Solutions: (1) log-sum-exp trick for softmax: log(Σexp(xᵢ)) = max(x) + log(Σexp(xᵢ - max(x))). (2) Use log probabilities instead of probabilities. (3) Gradient clipping for exploding gradients. (4) Mixed precision (FP16/BF16) requires loss scaling.

### Q30: How does attention relate to matrix multiplication?
**A**: Attention(Q,K,V) = softmax(QK^T/√d_k)·V. QK^T is a matrix multiplication producing attention scores (how much each query attends to each key). Softmax normalizes to probabilities. Multiplying by V produces a weighted sum of values. The entire attention mechanism is just matrix multiplications with a softmax — which is why GPUs are so efficient at it.

---

## 5. Day-to-Day Work Applications

### As an AI/LLM Engineer

**Linear Algebra Every Day**: Every embedding lookup, attention computation, and linear layer IS matrix multiplication. Understanding dimensions prevents shape mismatch errors. LoRA's low-rank update is SVD-inspired. Cosine similarity drives your RAG retrieval.

**Calculus for Training**: Every optimizer step uses gradients (chain rule). Understanding learning rate schedules, gradient clipping, and loss functions requires calculus intuition. Debugging training issues ("loss plateaued," "gradients exploded") needs calculus understanding.

**Probability for Evaluation**: A/B testing prompt changes. Calculating confidence intervals for model metrics. Understanding sampling strategies (temperature, top-p) for generation. Bayesian optimization for hyperparameter search.

**In System Design Interviews**: "How would you evaluate this model?" → statistical testing, confidence intervals. "Why does this loss function work?" → cross-entropy, MLE. "How does attention work?" → matrix multiplication, softmax, projections.

---

## 6. Resources

### Excel Curriculum Links
- Linear Algebra: https://www.youtube.com/playlist?list=PLZHQObOWTQDPD3MizzM2xVFitgF8hE_ab (3Blue1Brown)
- Calculus: https://www.youtube.com/playlist?list=PLZHQObOWTQDMsr9K-rj53DwVRMYO3t5Yr (3Blue1Brown)
- Probability & Statistics: https://www.youtube.com/watch?v=KbB0FjPg0mw
- Mathematics for ML Book: https://mml-book.github.io/
- Khan Academy Linear Algebra: https://www.khanacademy.org/math/linear-algebra
- StatQuest: https://www.youtube.com/c/joshstarmer
