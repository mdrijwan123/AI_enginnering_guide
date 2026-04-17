# Deep Learning Foundations — Complete Study Guide

> **Excel Curriculum Coverage**: Neural Network Basics, Backpropagation, CNN Architecture, RNN & LSTM, Transfer Learning, PyTorch/TensorFlow, Regularization, Optimization Techniques, Model Architectures
> **Interview Focus**: Fundamentals → architectures → training techniques → practical implementation
> **Day-to-Day**: Understanding DL internals is essential for fine-tuning, debugging, and architecting LLM systems

---

## Table of Contents
1. [Neural Network Basics](#1-neural-network-basics)
2. [Activation Functions](#2-activation-functions)
3. [Loss Functions](#3-loss-functions)
4. [Backpropagation](#4-backpropagation)
5. [Optimization Techniques](#5-optimization-techniques)
6. [Regularization](#6-regularization)
7. [CNN Architecture](#7-cnn-architecture)
8. [Classic Model Architectures](#8-classic-model-architectures)
9. [Transfer Learning](#9-transfer-learning)
10. [PyTorch Practical Guide](#10-pytorch-practical-guide)
11. [Interview Questions (40 Q&As)](#11-interview-questions)
12. [Day-to-Day Work Applications](#12-day-to-day-work-applications)
13. [Resources](#13-resources)

---

## 1. Neural Network Basics

> 💡 **ELI5 (Explain Like I'm 5):**
> Imagine trying to guess the price of a house. You might guess based on size, location, and age. A neural network is like a huge team of guessers. The first group looks at basic things (is it big?). They pass their answers to the next group, who combine them into more complex thoughts (large house + good location = high price). The system starts out guessing randomly, but every time it's wrong, we correct the guessers slightly until they're perfectly tuned.

> 📖 **Big picture:** A neural network is a function approximator inspired loosely by neurons in the brain. The key insight: by stacking layers of "neurons" (each just a weighted sum followed by a non-linear squash), you can approximate *any* function given enough data and the right architecture.
>
> **The weight tuning process:** Initially, all weights are random. The network makes terrible predictions. You measure how wrong it is (the *loss*), then you figure out which weights to nudge in which direction to reduce the loss (*backpropagation*), and nudge them (*gradient descent*). Repeat millions of times on millions of examples. Gradually the weights shift towards configurations that make good predictions. That’s training.
>
> **Why does this work for LLMs?** The transformer architecture is a neural network. GPT-4 is "just" a very deep neural network with 96 layers and 175 billion parameters. Every concept here — activations, loss, backprop, optimisers, regularisation — applies directly to how LLMs are trained.

### The Perceptron

The simplest neural unit — a single linear classifier:

$$y = \text{step}(w_1 x_1 + w_2 x_2 + \ldots + w_n x_n + b)$$

```python
import numpy as np

class Perceptron:
    def __init__(self, n_features, lr=0.01):
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.lr = lr
    
    def predict(self, x):
        return 1 if np.dot(self.weights, x) + self.bias > 0 else 0
    
    def train(self, X, y, epochs=100):
        for _ in range(epochs):
            for xi, yi in zip(X, y):
                pred = self.predict(xi)
                error = yi - pred
                self.weights += self.lr * error * xi
                self.bias += self.lr * error

# XOR problem — perceptron CANNOT solve this (not linearly separable)
# This limitation motivated multi-layer networks
```

### Multi-Layer Perceptron (MLP)

Stack layers of neurons to learn non-linear functions:

```
Input Layer → Hidden Layer 1 → Hidden Layer 2 → Output Layer
   (n)          (128)             (64)            (classes)
```

```python
import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

model = MLP(input_dim=784, hidden_dims=[256, 128], output_dim=10)
# Total params: 784*256 + 256 + 256*128 + 128 + 128*10 + 10 = ~235K
```

### Universal Approximation Theorem
A neural network with a single hidden layer of sufficient width can approximate any continuous function. However, deeper networks are exponentially more parameter-efficient than wider ones.

---

## 2. Activation Functions

> 📖 **Why they exist:** Without non-linear activations, stacking linear layers collapses into a single linear layer. Non-linearity lets networks model complex functions. The choice of activation function affects training stability, and several generations of activations have been developed to fix specific problems:
> - **Sigmoid/Tanh:** Early choices, but cause *vanishing gradients* in deep networks (derivatives get tiny, early layers don’t learn)
> - **ReLU:** Solved vanishing gradients for most networks; hugely popular
> - **GELU/SiLU:** What modern LLMs use; smoother than ReLU, better in practice at scale

### Why Non-linear Activations?
Without non-linear activations, stacking linear layers is equivalent to a single linear layer:
$f(x) = W_2(W_1 x + b_1) + b_2 = (W_2 W_1)x + (W_2 b_1 + b_2) = W'x + b'$

### Common Activation Functions

| Function | Formula | Range | Derivative | Used In |
|----------|---------|-------|------------|---------|
| **Sigmoid** | $\sigma(x) = \frac{1}{1+e^{-x}}$ | (0, 1) | $\sigma(1-\sigma)$ | Binary classification output |
| **Tanh** | $\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}}$ | (-1, 1) | $1 - \tanh^2(x)$ | RNNs, output normalization |
| **ReLU** | $\max(0, x)$ | [0, ∞) | 0 or 1 | Default for hidden layers |
| **Leaky ReLU** | $\max(0.01x, x)$ | (-∞, ∞) | 0.01 or 1 | Avoids dying ReLU |
| **GELU** | $x \cdot \Phi(x)$ | (-0.17, ∞) | Complex | BERT, GPT, modern LLMs |
| **SwiGLU** | $\text{Swish}(xW) \odot (xV)$ | Varies | Complex | LLaMA, Mistral, modern LLMs |
| **Softmax** | $\frac{e^{x_i}}{\sum_j e^{x_j}}$ | (0, 1), sums to 1 | — | Multi-class output, attention |

```python
import torch
import torch.nn.functional as F

x = torch.tensor([-2.0, -1.0, 0.0, 1.0, 2.0])

# ReLU: kills negative values
print(F.relu(x))          # [0, 0, 0, 1, 2]

# GELU: smooth approximation of ReLU (used in Transformers)
print(F.gelu(x))          # [-0.0454, -0.1588, 0, 0.8413, 1.9546]

# Sigmoid: squashes to (0, 1)
print(torch.sigmoid(x))   # [0.1192, 0.2689, 0.5, 0.7311, 0.8808]

# Softmax: probability distribution
logits = torch.tensor([2.0, 1.0, 0.1])
print(F.softmax(logits, dim=0))  # [0.6590, 0.2424, 0.0986]
```

### Why GELU and SwiGLU in Modern LLMs?

- **GELU**: Smooth activation that allows small negative values through. Used in BERT, GPT-2, RoBERTa. Outperforms ReLU in language tasks.
- **SwiGLU**: Gated Linear Unit variant used in LLaMA, Mistral, PaLM. Combines gating mechanism with Swish activation. Requires 50% more parameters per FFN layer but gets better perplexity per parameter.

---

## 3. Loss Functions

### For Classification

```python
# Binary Cross-Entropy (sigmoid output)
bce_loss = nn.BCEWithLogitsLoss()  # Includes sigmoid internally
# L = -[y·log(σ(x)) + (1-y)·log(1-σ(x))]

# Cross-Entropy (softmax output for multi-class)
ce_loss = nn.CrossEntropyLoss()  # Includes softmax internally
# L = -∑ y_i · log(softmax(x_i))

# Focal Loss (for class imbalance)
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # Predicted probability of correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()
```

### For Regression

```python
# Mean Squared Error (L2 loss)
mse = nn.MSELoss()   # L = (1/n) · ∑(y - ŷ)²

# Mean Absolute Error (L1 loss) — robust to outliers
mae = nn.L1Loss()    # L = (1/n) · ∑|y - ŷ|

# Huber Loss (smooth L1) — best of both worlds
huber = nn.SmoothL1Loss()  # L1 when |error| > 1, L2 otherwise
```

### For LLMs

```python
# Next-token prediction: Cross-entropy over vocabulary
# logits: (batch, seq_len, vocab_size)
# targets: (batch, seq_len)
loss = F.cross_entropy(
    logits.view(-1, vocab_size),  # (batch*seq_len, vocab_size)
    targets.view(-1),              # (batch*seq_len,)
    ignore_index=-100              # Ignore padding tokens
)

# KL Divergence (used in distillation, RLHF)
kl_loss = nn.KLDivLoss(reduction='batchmean')
# KL(P || Q) = ∑ P(x) · log(P(x)/Q(x))
```

---

## 4. Backpropagation

> 💡 **ELI5 (Explain Like I'm 5):** 
> Imagine a restaurant makes a terrible cake. The head chef tastes test it (calculates the error) and yells at the baker: "Too salty!" The baker then yells at the prep cook: "You added too much salt!" The prep cook yells at the supplier: "Your salt shakers are too big!" 
> Backpropagation is exactly this: blaming the error on the people (weights) responsible, tracing backward from the final result to the very beginning, so everyone knows how to adjust for the next cake.

> 📖 **Plain English:** Backpropagation is how neural networks learn from mistakes. After a forward pass produces a prediction and we measure the error (loss), we need to know: *which weights caused the error, and by how much?* We trace the error backwards through the network, layer by layer, using the **chain rule** from calculus. Each weight gets a gradient — a signed number saying "increase this weight → loss goes up/down by this much." Then gradient descent nudges every weight in the direction that reduces loss.
>
> **The human analogy:** Your boss tells you the project failed. You figure out your role: "I made decisions X and Y. X was good, Y was bad. I’ll change how I make decision-Y-type choices in future." Backprop is the mathematical equivalent of this attribution process across millions of parameters.

### Chain Rule — The Foundation

Backpropagation is just the chain rule applied systematically:

$$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial z} \cdot \frac{\partial z}{\partial w}$$

### Worked Example: 2-Layer Network

```
Forward: x → [z1 = W1·x + b1] → [a1 = ReLU(z1)] → [z2 = W2·a1 + b2] → [ŷ = σ(z2)] → L = BCE(ŷ, y)

Backward (chain rule):
∂L/∂z2 = ŷ - y                           (error signal)
∂L/∂W2 = (∂L/∂z2) · a1^T                  (gradient for W2)
∂L/∂b2 = ∂L/∂z2                           (gradient for b2)
∂L/∂a1 = W2^T · (∂L/∂z2)                  (propagate error back)
∂L/∂z1 = (∂L/∂a1) * ReLU'(z1)            (through activation)
∂L/∂W1 = (∂L/∂z1) · x^T                   (gradient for W1)
∂L/∂b1 = ∂L/∂z1                           (gradient for b1)
```

### Implementation from Scratch

```python
import numpy as np

class NeuralNetwork:
    """2-layer neural network with backpropagation from scratch."""
    
    def __init__(self, input_size, hidden_size, output_size, lr=0.01):
        # Xavier initialization
        self.W1 = np.random.randn(input_size, hidden_size) * np.sqrt(2.0 / input_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.randn(hidden_size, output_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros((1, output_size))
        self.lr = lr
    
    def relu(self, z):
        return np.maximum(0, z)
    
    def relu_derivative(self, z):
        return (z > 0).astype(float)
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    
    def forward(self, X):
        """Forward pass — store intermediates for backprop."""
        self.z1 = X @ self.W1 + self.b1          # Linear 1
        self.a1 = self.relu(self.z1)               # Activation 1
        self.z2 = self.a1 @ self.W2 + self.b2     # Linear 2
        self.a2 = self.sigmoid(self.z2)            # Output
        return self.a2
    
    def backward(self, X, y):
        """Backward pass — compute gradients via chain rule."""
        m = X.shape[0]
        
        # Output layer gradients
        dz2 = self.a2 - y                          # ∂L/∂z2 = ŷ - y
        dW2 = (self.a1.T @ dz2) / m               # ∂L/∂W2
        db2 = np.sum(dz2, axis=0, keepdims=True) / m
        
        # Hidden layer gradients (chain rule through ReLU)
        da1 = dz2 @ self.W2.T                     # ∂L/∂a1
        dz1 = da1 * self.relu_derivative(self.z1)  # ∂L/∂z1
        dW1 = (X.T @ dz1) / m                     # ∂L/∂W1
        db1 = np.sum(dz1, axis=0, keepdims=True) / m
        
        # Update weights
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
    
    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y)
            if epoch % 100 == 0:
                loss = -np.mean(y * np.log(output + 1e-8) + (1-y) * np.log(1-output + 1e-8))
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

# Solve XOR (impossible for single perceptron)
X = np.array([[0,0], [0,1], [1,0], [1,1]])
y = np.array([[0], [1], [1], [0]])
nn = NeuralNetwork(2, 4, 1, lr=0.5)
nn.train(X, y, epochs=5000)
print(nn.forward(X))  # Close to [[0], [1], [1], [0]]
```

### Computational Graphs & Autograd

PyTorch builds a computational graph dynamically and computes gradients automatically:

```python
import torch

x = torch.tensor(2.0, requires_grad=True)
y = torch.tensor(3.0, requires_grad=True)

# Forward: build computational graph
z = x**2 + 3*x*y + y**2  # z = 4 + 18 + 9 = 31

# Backward: compute gradients via chain rule
z.backward()

print(x.grad)  # dz/dx = 2x + 3y = 4 + 9 = 13
print(y.grad)  # dz/dy = 3x + 2y = 6 + 6 = 12
```

---

## 5. Optimization Techniques

> 📖 **Big picture:** Once you have gradients from backpropagation, you have to decide how to use them to update weights. Plain gradient descent says "move every weight in the direction of its negative gradient." But this naive approach has problems: it oscillates, gets stuck in saddle points, and is slow if you have to set the learning rate carefully.
>
> **Adam is the default for LLMs:** Adaptive Moment Estimation (Adam) combines momentum (remember past gradients to smooth updates) with adaptive learning rates (make small updates for parameters that have big gradients, and big updates for parameters with small gradients). It’s the go-to optimiser for almost all modern deep learning including transformers.

### SGD (Stochastic Gradient Descent)

$$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)$$

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
```

**Problems**: Oscillates in steep dimensions, slow convergence, gets stuck in saddle points.

### SGD with Momentum

Add "velocity" to smooth updates:

$$v_t = \beta v_{t-1} + \nabla L(\theta_t)$$
$$\theta_{t+1} = \theta_t - \eta v_t$$

```python
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

### Adam (Adaptive Moment Estimation)

Combines momentum with per-parameter learning rates:

$$m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{(first moment — mean)}$$
$$v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{(second moment — variance)}$$
$$\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t} \quad \text{(bias correction)}$$
$$\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$$

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-8)
```

### AdamW (Weight Decay Decoupled)

Adam has a subtle bug: weight decay interacts with adaptive learning rates. AdamW fixes this by decoupling weight decay from the gradient-based update:

$$\theta_{t+1} = \theta_t - \eta(\frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_t)$$

```python
# THE standard optimizer for Transformers/LLMs
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
```

### Learning Rate Scheduling

```python
from torch.optim.lr_scheduler import (
    StepLR, CosineAnnealingLR, OneCycleLR, 
    CosineAnnealingWarmRestarts
)

# --- Warm-up + Cosine Decay (standard for LLM training) ---
def get_lr_scheduler(optimizer, warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps  # Linear warmup
        # Cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# Example usage
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
scheduler = get_lr_scheduler(optimizer, warmup_steps=1000, total_steps=100000)

# In training loop:
for step in range(total_steps):
    loss = train_step()
    loss.backward()
    optimizer.step()
    scheduler.step()
    optimizer.zero_grad()

# --- Cosine Annealing ---
scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)

# --- One Cycle Policy (fastest convergence) ---
scheduler = OneCycleLR(optimizer, max_lr=1e-3, total_steps=10000)
```

### Comparison Table

| Optimizer | When to Use | Key Property |
|-----------|------------|--------------|
| **SGD** | Simple problems, lots of data | Can generalize better with tuning |
| **SGD + Momentum** | CNNs, computer vision | Smooths oscillations |
| **Adam** | General default | Fast convergence, less tuning |
| **AdamW** | **Transformers, LLMs (standard)** | Proper weight decay |
| **Adafactor** | Very large models (memory savings) | Factored second moments |

---

## 6. Regularization

> 💡 **ELI5 (Explain Like I'm 5):** 
> If you study for a math test by memorising *exactly* the 10 practice questions, you'll fail the real test because the numbers changed. You memorised instead of learning the rules. Regularization is like a teacher randomly changing the practice numbers while you study, or forcing you to solve equations with one hand tied behind your back (Dropout). It forces the AI to learn the *actual rules* instead of just memorising the training data.

> 📖 **The problem:** A neural network with millions of parameters can *memorise* the training data rather than learning general patterns. It gets 99% accuracy on training data but fails on new examples. This is **overfitting**. Regularisation is a collection of techniques that push the model towards simpler, more general solutions.
>
> - **Dropout:** Randomly disable 20-50% of neurons during each training step. Prevents any single neuron from becoming essential, forces redundant representations. Disabled at inference time.
> - **Weight decay (L2):** Penalises large weights, encouraging the model to spread knowledge across many weights rather than concentrating it.
> - **Batch Normalisation:** Normalises layer outputs to have zero mean and unit variance, allowing higher learning rates and more stable training.

### Dropout

Randomly zero out neurons during training — prevents co-adaptation:

```python
class DropoutDemo(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.dropout = nn.Dropout(p=0.5)  # Drop 50% of neurons
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  # Only active during model.train()
        return self.fc2(x)

model = DropoutDemo()
model.train()   # Dropout active
model.eval()    # Dropout disabled (scale outputs by 1-p)
```

### Batch Normalization

Normalize activations within each mini-batch — stabilizes and accelerates training:

$$\hat{x}_i = \frac{x_i - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$$
$$y_i = \gamma \hat{x}_i + \beta$$

Where $\mu_B$ and $\sigma_B$ are batch statistics, and $\gamma$, $\beta$ are learnable.

```python
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)  # Normalize per channel
        self.relu = nn.ReLU()
    
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
# Order: Conv → BatchNorm → Activation

# Layer Normalization (used in Transformers instead of BatchNorm)
layer_norm = nn.LayerNorm(768)  # Normalize across features, not batch
# Preferred in Transformers because it's independent of batch size
```

### BatchNorm vs LayerNorm

| Feature | BatchNorm | LayerNorm |
|---------|-----------|-----------|
| Normalizes across | Batch dimension | Feature dimension |
| Depends on batch size | ✅ Yes | ❌ No |
| Used in | CNNs | **Transformers, RNNs** |
| At inference | Uses running stats | Same as training |
| Sequence tasks | ❌ Problematic | ✅ Standard |

### Weight Decay (L2 Regularization)

Add penalty for large weights to the loss:

$$L_{\text{reg}} = L + \lambda \sum_i w_i^2$$

```python
# In PyTorch, applied via optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
```

### Early Stopping

```python
class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
    
    def should_stop(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience
```

### Data Augmentation (for images)

```python
from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

---

## 7. CNN Architecture

### Core Components

```
Input Image → [Conv → ReLU → Pool] × N → Flatten → FC → Output
```

**Convolution**: Sliding filter extracts local features
- **Kernel/Filter**: Small matrix (3×3, 5×5) that slides over input
- **Stride**: Step size of the sliding window
- **Padding**: Add zeros around border to control output size
- **Feature Map**: Output of applying one filter to the input

```python
class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        # Input: (batch, 3, 32, 32)  [3 = RGB channels]
        self.features = nn.Sequential(
            # Conv1: 3 → 32 channels, 3x3 kernel
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # → (batch, 32, 32, 32)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                           # → (batch, 32, 16, 16)
            
            # Conv2: 32 → 64 channels
            nn.Conv2d(32, 64, kernel_size=3, padding=1),  # → (batch, 64, 16, 16)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),                            # → (batch, 64, 8, 8)
            
            # Conv3: 64 → 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),  # → (batch, 128, 8, 8)
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),                  # → (batch, 128, 1, 1)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                    # → (batch, 128)
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)      # → (batch, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)
```

### Output Size Formula

$$H_{out} = \lfloor \frac{H_{in} + 2P - K}{S} \rfloor + 1$$

Where $H_{in}$ = input height, $P$ = padding, $K$ = kernel size, $S$ = stride.

### Pooling Layers

| Type | Purpose | How |
|------|---------|-----|
| **Max Pooling** | Keep strongest features | Take max value in window |
| **Average Pooling** | Smooth features | Take average in window |
| **Global Average Pooling** | Replace FC layers | Average entire feature map to one value |
| **Adaptive Pooling** | Fixed output size | Auto-compute kernel size |

### Receptive Field

The area of the original input that influences a particular feature:
- Layer 1 (3×3 conv): receptive field = 3×3
- Layer 2 (3×3 conv): receptive field = 5×5
- Layer 3 (3×3 conv): receptive field = 7×7
- With pooling: grows faster

Deeper networks = larger receptive fields = can recognize more complex patterns

### 1×1 Convolutions
Not spatial filtering — serves as **channel-wise linear transformation**:
- Reduce/increase channels (dimensionality reduction)
- Add non-linearity with minimal parameters
- Used extensively in Inception, ResNet bottlenecks

---

## 8. Classic Model Architectures

### LeNet-5 (1998) — The Pioneer
First successful CNN for digit recognition.
```
Input(32×32) → Conv(5×5) → Pool → Conv(5×5) → Pool → FC → FC → Output
```

### VGGNet (2014) — Depth with Simplicity
All 3×3 convolutions, very deep (16/19 layers). Showed depth matters.
```
Input → [3×3 Conv → 3×3 Conv → Pool] × 5 → FC → FC → Output
138M parameters — very large!
```

### ResNet (2015) — Skip Connections

The breakthrough: **residual connections** allow training very deep networks (50, 101, 152 layers).

$$F(x) + x \quad \text{(skip/shortcut connection)}$$

```python
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
    
    def forward(self, x):
        residual = x                          # Save input
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual                        # Skip connection!
        return F.relu(out)

# Bottleneck Block (used in ResNet-50+)
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, bottleneck_channels, 1)      # 1×1 reduce
        self.bn1 = nn.BatchNorm2d(bottleneck_channels)
        self.conv2 = nn.Conv2d(bottleneck_channels, bottleneck_channels, 3, padding=1)  # 3×3
        self.bn2 = nn.BatchNorm2d(bottleneck_channels)
        self.conv3 = nn.Conv2d(bottleneck_channels, out_channels, 1)     # 1×1 expand
        self.bn3 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return F.relu(out + self.shortcut(x))
```

**Why ResNets work**: Skip connections create "gradient highways" — gradients can flow directly through shortcuts, solving the vanishing gradient problem in very deep networks.

### Inception/GoogLeNet (2014) — Multi-Scale Features

Process input at multiple scales simultaneously:

```python
class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_1x1, out_3x3, out_5x5, out_pool):
        super().__init__()
        # Branch 1: 1×1 conv
        self.branch1 = nn.Conv2d(in_channels, out_1x1, 1)
        
        # Branch 2: 1×1 → 3×3 conv
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_3x3 // 2, 1),
            nn.Conv2d(out_3x3 // 2, out_3x3, 3, padding=1)
        )
        
        # Branch 3: 1×1 → 5×5 conv
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_5x5 // 4, 1),
            nn.Conv2d(out_5x5 // 4, out_5x5, 5, padding=2)
        )
        
        # Branch 4: MaxPool → 1×1 conv
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_pool, 1)
        )
    
    def forward(self, x):
        return torch.cat([
            self.branch1(x), self.branch2(x),
            self.branch3(x), self.branch4(x)
        ], dim=1)
```

### EfficientNet (2019) — Compound Scaling

Systematically scale width, depth, and resolution together using a compound coefficient $\phi$:

$$\text{depth}: d = \alpha^\phi, \quad \text{width}: w = \beta^\phi, \quad \text{resolution}: r = \gamma^\phi$$

```python
# Using pretrained EfficientNet
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
model = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
# B0: 5.3M params, B7: 66M params (same architecture, different scale)
```

### Architecture Comparison

| Model | Year | Depth | Params | Top-5 Error | Key Innovation |
|-------|------|-------|--------|-------------|----------------|
| VGG-16 | 2014 | 16 | 138M | 7.3% | Depth with 3×3 convs |
| GoogLeNet | 2014 | 22 | 6.8M | 6.7% | Multi-scale (Inception) |
| ResNet-50 | 2015 | 50 | 25M | 5.3% | Skip connections |
| ResNet-152 | 2015 | 152 | 60M | 4.5% | Very deep with residuals |
| EfficientNet-B7 | 2019 | 66M | 66M | 2.9% | Compound scaling |

---

## 9. Transfer Learning

### The Concept
Use a model pre-trained on a large dataset (ImageNet, large text corpus) and adapt it to your task.

```
Pre-trained Model (ImageNet, 1.2M images, 1000 classes)
         ↓
    Feature Extractor (early layers: edges, textures)
         ↓
    Fine-tune or Replace classifier head
         ↓
    Your Task (e.g., 100 medical images, 5 classes)
```

### Strategies

**1. Feature Extraction** (freeze pretrained, train only new head):
```python
from torchvision.models import resnet50, ResNet50_Weights

model = resnet50(weights=ResNet50_Weights.DEFAULT)

# Freeze ALL pretrained layers
for param in model.parameters():
    param.requires_grad = False

# Replace classifier head
model.fc = nn.Linear(model.fc.in_features, num_classes)
# Only model.fc has requires_grad=True
```

**2. Fine-tuning** (unfreeze some/all layers, train with small LR):
```python
model = resnet50(weights=ResNet50_Weights.DEFAULT)

# Freeze early layers (generic features)
for name, param in model.named_parameters():
    if 'layer4' not in name and 'fc' not in name:
        param.requires_grad = False

# Replace classifier
model.fc = nn.Linear(model.fc.in_features, num_classes)

# Use smaller LR for pretrained layers, larger for new head
optimizer = torch.optim.Adam([
    {'params': model.layer4.parameters(), 'lr': 1e-5},
    {'params': model.fc.parameters(), 'lr': 1e-3},
])
```

**3. Gradual Unfreezing** (unfreeze layers progressively):
```python
# Epoch 1-3: Train only FC head
# Epoch 4-6: Unfreeze layer4 + FC
# Epoch 7+: Unfreeze all layers
for epoch in range(num_epochs):
    if epoch == 4:
        for param in model.layer4.parameters():
            param.requires_grad = True
    if epoch == 7:
        for param in model.parameters():
            param.requires_grad = True
```

### When to Apply Transfer Learning

| Your Data | Your Task vs Pretrained | Strategy |
|-----------|------------------------|----------|
| Small, similar | Classification (similar domain) | Feature extraction |
| Small, different | Classification (different domain) | Feature extraction from earlier layers |
| Large, similar | Same domain, more specific | Fine-tune all layers |
| Large, different | Completely different | Train from scratch or fine-tune carefully |

---

## 10. PyTorch Practical Guide

### Tensors — The Foundation

```python
import torch

# Creation
x = torch.tensor([1, 2, 3])               # From list
x = torch.zeros(3, 4)                      # 3×4 zeros
x = torch.randn(3, 4)                      # Normal distribution
x = torch.arange(0, 10, 2)                 # [0, 2, 4, 6, 8]
x = torch.linspace(0, 1, 5)               # [0, 0.25, 0.5, 0.75, 1]

# Operations
a = torch.randn(3, 4)
b = torch.randn(3, 4)
c = a + b                                  # Element-wise add
d = a @ b.T                                # Matrix multiplication (3×3)
e = torch.matmul(a, b.T)                   # Same as @

# Reshaping
x = torch.randn(6, 4)
x.view(2, 3, 4)                            # Reshape (must be contiguous)
x.reshape(2, 3, 4)                         # Reshape (always works)
x.unsqueeze(0)                             # Add dim: (1, 6, 4)
x.squeeze()                                # Remove dim=1 dimensions

# GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
x = x.to(device)
```

### Dataset & DataLoader

```python
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx])
        }

dataset = TextDataset(texts, labels, tokenizer)
dataloader = DataLoader(
    dataset, 
    batch_size=32, 
    shuffle=True,           # Shuffle training data
    num_workers=4,          # Parallel data loading
    pin_memory=True,        # Faster GPU transfer
    drop_last=True          # Drop incomplete last batch
)
```

### Complete Training Loop

```python
def train_epoch(model, dataloader, optimizer, scheduler, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        # Move to device
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = F.cross_entropy(outputs, labels)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
        optimizer.step()
        scheduler.step()
        
        # Track metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(dataloader), correct / total

@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = F.cross_entropy(outputs, labels)
        
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)
    
    return total_loss / len(dataloader), correct / total

# Training loop
best_val_loss = float('inf')
for epoch in range(num_epochs):
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, scheduler, device)
    val_loss, val_acc = evaluate(model, val_loader, device)
    
    print(f"Epoch {epoch}: Train Loss={train_loss:.4f} Acc={train_acc:.4f} | Val Loss={val_loss:.4f} Acc={val_acc:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), 'best_model.pt')
```

### Autograd Deep Dive

```python
# Gradient accumulation (for effective larger batch sizes)
accumulation_steps = 4
optimizer.zero_grad()
for i, batch in enumerate(dataloader):
    loss = model(batch) / accumulation_steps  # Scale loss
    loss.backward()                            # Accumulate gradients
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# Mixed precision training (faster, less memory)
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

for batch in dataloader:
    optimizer.zero_grad()
    with autocast():
        output = model(batch)
        loss = criterion(output, target)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

---

## 11. Interview Questions

### Q1: What is the universal approximation theorem?
**A**: A neural network with a single hidden layer of sufficient width can approximate any continuous function on a compact subset of R^n to arbitrary precision. However, the required width may be exponentially large. In practice, deeper but narrower networks are more parameter-efficient and generalize better.

### Q2: Why do we need non-linear activation functions?
**A**: Without non-linearity, stacking linear layers is equivalent to a single linear layer (matrix multiplication is associative). Non-linear activations allow the network to model complex, non-linear decision boundaries. Without them, the network can only learn linear functions, which is insufficient for most real-world problems.

### Q3: Explain the dying ReLU problem.
**A**: If a neuron's pre-activation becomes negative for all inputs, ReLU outputs zero and the gradient is also zero. The neuron "dies" — its weights are never updated. This happens with high learning rates or large negative biases. Solutions: Leaky ReLU (small slope for negatives), PReLU (learned slope), ELU, or GELU.

### Q4: Why is GELU preferred over ReLU in Transformers?
**A**: GELU is smooth and differentiable everywhere (unlike ReLU's sharp corner at 0). It allows small negative values through (weighted by their probability under a Gaussian), providing a soft gating effect. Empirically, GELU gives better perplexity in language models. SwiGLU further improves on this with a gating mechanism.

### Q5: Derive backpropagation for a single neuron.
**A**: For $y = \sigma(wx + b)$ with MSE loss $L = (y - t)^2$:
$\frac{\partial L}{\partial w} = \frac{\partial L}{\partial y} \cdot \frac{\partial y}{\partial z} \cdot \frac{\partial z}{\partial w}$ where $z = wx + b$
$= 2(y-t) \cdot \sigma'(z) \cdot x$
$= 2(y-t) \cdot \sigma(z)(1-\sigma(z)) \cdot x$

### Q6: What is the difference between batch, mini-batch, and stochastic gradient descent?
**A**: Batch GD: Use entire dataset per update — stable but slow, high memory. SGD: Use single sample per update — noisy but fast, low memory. Mini-batch GD: Use a batch (32-512 samples) — best trade-off between stability and speed. Mini-batch is the practical default. "SGD" in deep learning usually means mini-batch SGD.

### Q7: Why does Adam work better than SGD for Transformers?
**A**: Transformers have sparse gradients (attention matrices). Adam's per-parameter adaptive learning rates handle this well — rarely updated parameters get larger effective learning rates. SGD uses the same LR for all parameters, which doesn't work well when some parameters need much larger updates than others. AdamW adds proper weight decay.

### Q8: Explain learning rate warmup. Why is it needed?
**A**: Initially, model parameters are random and gradients can be very large. Warmup starts with a very small LR and linearly increases to the target LR over a warmup period. This prevents early large updates from destabilizing training. Especially important for Transformer models and when using Adam (adaptive moments need time to calibrate).

### Q9: What is gradient clipping and when is it needed?
**A**: Cap gradient norms to a maximum value before the optimizer step. Prevents exploding gradients from destabilizing training. Two types: by value (clip each gradient independently) and by norm (scale entire gradient vector). Standard for RNNs/LSTMs and Transformers. Typical max norm: 1.0 or 5.0.

### Q10: Explain batch normalization step by step.
**A**: (1) Compute batch mean μ and variance σ² for each feature. (2) Normalize: (x - μ) / √(σ² + ε). (3) Scale and shift with learnable γ and β: y = γx̂ + β. During training: use batch statistics. During inference: use running average statistics collected during training. Benefits: faster convergence, allows higher LR, some regularization.

### Q11: Why do Transformers use LayerNorm instead of BatchNorm?
**A**: BatchNorm normalizes across the batch dimension — depends on batch size and breaks with variable-length sequences. LayerNorm normalizes across the feature dimension for each individual sample — independent of batch size. In language tasks with variable sequence lengths and small batches, LayerNorm is more stable and consistent.

### Q12: What is the difference between L1 and L2 regularization?
**A**: L1 ($\lvert w \rvert$): Drives weights exactly to zero → feature selection/sparsity. L2 ($w^2$): Shrinks weights toward zero but rarely exactly zero → smooth regularization. L1 creates sparse models (useful for feature selection). L2 creates smooth models (useful when all features matter). ElasticNet combines both: $\alpha\lvert w \rvert + (1-\alpha)w^2$.

### Q13: What is a convolution mathematically?
**A**: Output[i,j] = Σ_m Σ_n Input[i+m, j+n] × Kernel[m, n]. The kernel slides over the input, computing element-wise multiplication and summing. In CNNs, the kernel weights are learned via backpropagation. Each kernel detects a specific feature (edges, textures, shapes). Multiple kernels = multiple feature maps per layer.

### Q14: Why do small kernels (3×3) work better than large ones?
**A**: Two 3×3 convolutions have the same receptive field as one 5×5 (RF=5×5) but with fewer parameters (2×9=18 vs 25) and more non-linearity (two ReLU activations vs one). Three 3×3 convolutions = one 7×7 receptive field. VGGNet demonstrated this principle. Modern architectures use 3×3 almost exclusively.

### Q15: Explain ResNet's skip connections and why they help.
**A**: Instead of learning H(x), ResNets learn the residual F(x) = H(x) - x, outputting F(x) + x. Benefits: (1) If the identity mapping is optimal, F(x) just needs to be zero (easier than learning identity). (2) Gradients flow directly through skip connections, solving vanishing gradients in deep networks. (3) Enables training networks with 100+ layers.

### Q16: What is depthwise separable convolution?
**A**: Splits standard convolution into: (1) Depthwise conv — one filter per input channel. (2) Pointwise conv — 1×1 conv to combine channels. Standard: C_in × C_out × K × K params. Separable: C_in × K × K + C_in × C_out params. For K=3, C_in=C_out=256: 589K vs 66K params (9× reduction). Used in MobileNet, EfficientNet.

### Q17: How does max pooling differ from stride convolution for downsampling?
**A**: Max pooling: Takes the maximum in each window. No learnable parameters. Provides some translation invariance. Stride convolution: Use stride > 1 in convolution layer. Has learnable parameters. Can learn what to keep. Modern architectures (ResNet-v2, EfficientNet) often prefer stride convolution as it's more expressive.

### Q18: Explain transfer learning intuition. Why does it work?
**A**: Early CNN layers learn generic features (edges, colors, textures) that are universal. Deeper layers learn task-specific features. Pre-trained models transfer generic features to new tasks, requiring less data to learn the task-specific parts. Works because visual/language features are hierarchical and many are shared across tasks.

### Q19: When would fine-tuning hurt performance?
**A**: (1) Very small dataset with very different domain — catastrophic forgetting. (2) Pre-trained model is much larger than needed — overfitting. (3) Source and target domains are fundamentally different (medical X-rays vs satellite imagery). (4) High learning rate destroys pre-trained features. Solutions: lower LR, freeze more layers, use feature extraction instead.

### Q20: What is mixed precision training?
**A**: Use FP16 for forward/backward passes (faster, less memory) and FP32 for weight updates (maintain precision). Loss scaling prevents gradient underflow in FP16. Benefits: ~2× speedup on modern GPUs, ~50% memory reduction. Essential for training large models. torch.cuda.amp makes it easy. BF16 (brain floating point) is increasingly preferred.

### Q21: Explain gradient accumulation and when to use it.
**A**: Instead of updating weights every batch, accumulate gradients over multiple batches and then update. Effective batch size = batch_size × accumulation_steps. Use when: (1) GPU memory can't fit desired batch size. (2) You need large effective batch sizes for training stability. Must scale learning rate accordingly (larger effective batch = higher LR).

### Q22: What is the difference between model.train() and model.eval() in PyTorch?
**A**: model.train(): Enables dropout (randomly zeros neurons), uses batch statistics for BatchNorm. model.eval(): Disables dropout (uses all neurons, scaled by 1-p), uses running statistics for BatchNorm. Critical in production — forgetting model.eval() causes inconsistent predictions. Also needed for torch.no_grad() context for inference.

### Q23: How does 1×1 convolution work and why is it useful?
**A**: 1×1 convolution operates on each spatial position independently, mixing across channels. It's equivalent to a fully connected layer applied to each pixel. Uses: (1) Channel reduction (128→32 channels before expensive 3×3 conv = bottleneck). (2) Adding non-linearity. (3) Cross-channel interaction. Key component of Inception modules and ResNet bottleneck blocks.

### Q24: What is global average pooling and why replaced FC layers?
**A**: Instead of flattening feature maps and using large FC layers, take the spatial average of each feature map to get one value per channel. Benefits: (1) Fewer parameters (no FC weights). (2) Less overfitting. (3) Works with any input size. (4) More interpretable — each feature map corresponds to a class confidence. Used in GoogLeNet onwards.

### Q25: Explain the concept of receptive field.
**A**: The area of the original input image that influences a single neuron's activation. Grows with depth: each 3×3 conv layer adds 2 to the RF. Pooling/stride multiplies the growth rate. ResNet-50 has RF > 400×400 pixels. Important because the network can only detect features within its RF — if RF is smaller than the object, it can't recognize it properly.

### Q26: What is the difference between transposed convolution and upsampling?
**A**: Upsampling: Nearest-neighbor or bilinear interpolation (no learnable parameters). Transposed convolution (deconvolution): Learns the upsampling operation, swapping the forward and backward passes of a regular convolution. Used in segmentation (U-Net), GANs (generators), autoencoders (decoders). Can produce checkerboard artifacts.

### Q27: Explain weight initialization. Why does it matter?
**A**: Bad initialization causes vanishing (if weights too small) or exploding (if too large) activations. Xavier/Glorot init: scale by $\sqrt{2/(\text{fan\_in} + \text{fan\_out})}$, good for sigmoid/tanh. Kaiming/He init: scale by $\sqrt{2/\text{fan\_in}}$, good for ReLU. PyTorch uses Kaiming by default for Conv/Linear layers.

### Q28: What is the vanishing gradient problem and how do modern architectures handle it?
**A**: Deep networks: gradients get exponentially smaller in early layers (chain rule multiplies many small numbers). Solutions: (1) ReLU activation (gradient is 0 or 1, not shrinking). (2) Skip connections (ResNet — gradient highway). (3) Batch/Layer normalization (keeps activations in good range). (4) Better initialization (Kaiming/He). (5) Gradient clipping for RNNs.

### Q29: Explain knowledge distillation.
**A**: Train a small "student" model to mimic a large "teacher" model. Loss = α × KL(teacher_logits, student_logits) + (1-α) × CE(true_labels, student_logits). The teacher's soft probabilities contain "dark knowledge" — relative similarities between classes (<0.1 for "cat" vs 0.01 for "car" tells the student these classes are somewhat related). DistilBERT, TinyBERT use this.

### Q30: What is the difference between data parallelism and model parallelism?
**A**: Data parallelism: Same model on multiple GPUs, each processing different data batches. Gradients are averaged across GPUs. Simple, scales well. Model parallelism: Split the model across GPUs (different layers or different parts of layers). Needed when model doesn't fit on one GPU. Can be pipeline parallel (split by layers) or tensor parallel (split within layers).

### Q31: Explain the bias-variance tradeoff in the context of neural networks.
**A**: High bias (underfitting): Model too simple — increase depth/width, train longer, reduce regularization. High variance (overfitting): Model memorizes training data — add dropout, weight decay, data augmentation, early stopping, reduce model size. Deep learning typically operates in a high-variance regime, so regularization is crucial.

### Q32: What is curriculum learning?
**A**: Train on easy examples first, then gradually increase difficulty. Similar to how humans learn. Benefits: faster convergence, better final performance on some tasks. Example: Sort training data by length (short sequences first) or by loss (low-loss examples first). Used in NMT and some LLM training procedures.

### Q33: How do you debug a neural network that isn't training?
**A**: (1) Check loss: Is it decreasing? NaN? (2) Overfit on a single batch first — if it can't memorize one batch, there's a bug. (3) Check learning rate (try 1e-3, 1e-4). (4) Check data pipeline (visualize inputs, check labels). (5) Check gradient norms (vanishing or exploding?). (6) Simplify model and add complexity incrementally. (7) Use gradient checking (numeric vs analytic).

### Q34: What are attention maps in CNNs?
**A**: Visualizations showing which parts of the input the model focuses on. Methods: Grad-CAM (gradient-weighted class activation maps), Attention Rollout (for Vision Transformers). Use the gradients flowing back to the last conv layer to weight the feature maps. Useful for debugging and interpretability — verify the model looks at the right features.

### Q35: Explain the lottery ticket hypothesis.
**A**: Dense neural networks contain sparse subnetworks ("winning tickets") that, when trained in isolation from their original initialization, reach comparable accuracy. Implication: you can prune 90%+ of weights without losing performance, but you need the right initial weights. Supports structured pruning as a compression technique.

### Q36: What is model quantization?
**A**: Reduce precision of model weights/activations from FP32 to INT8/INT4. Post-training quantization: Quantize after training (fast, slight accuracy loss). Quantization-aware training: Train with quantization simulation (slower, better accuracy). For LLMs: GGUF, GPTQ, AWQ, bitsandbytes (NF4/QLoRA). Typical: 4× model size reduction with <1% accuracy loss.

### Q37: Explain the difference between pre-activation and post-activation ResNet.
**A**: Post-activation (original): Conv → BN → ReLU → Conv → BN → Add → ReLU. Pre-activation (v2, improved): BN → ReLU → Conv → BN → ReLU → Conv → Add. Pre-activation is better because the identity path is completely clear (no BN or ReLU on the shortcut). This improves gradient flow and generalization.

### Q38: What is feature pyramid network (FPN)?
**A**: Combines features from multiple scales (early layers: high-res, low-level; later layers: low-res, high-level). Uses a top-down pathway with lateral connections to create strong features at all scales. Essential for object detection (detecting both small and large objects). Used in Faster R-CNN, Mask R-CNN, RetinaNet.

### Q39: How does Vision Transformer (ViT) work?
**A**: Split image into fixed-size patches (16×16). Flatten each patch, project to embedding dimension. Add position embeddings. Prepend a [CLS] token. Feed through standard Transformer encoder. Use [CLS] output for classification. Key insight: attention is all you need, even for images. Requires more data than CNNs but scales better.

### Q40: What's the relationship between CNNs and Transformers for vision?
**A**: CNNs have inductive biases (locality, translation equivariance) that help with limited data. Transformers are more flexible but need more data. Hybrid approaches: use CNN features as input to Transformer (ViT-Hybrid), or add convolutions to Transformers (CoAtNet). For LLM engineers: understanding both is key for multimodal models (CLIP, LLaVA).

---

## 12. Day-to-Day Work Applications

### As an AI/LLM Engineer

**Understanding Model Training & Debugging**:
- Diagnosing why fine-tuning diverges (check LR, gradient norms, loss curves)
- Choosing optimizer settings: AdamW with warmup + cosine decay is standard for LLMs
- Mixed precision (FP16/BF16) for faster training and lower memory
- Gradient accumulation for effective batch sizes when GPU-limited

**Transfer Learning in LLM Work**:
- Every fine-tuning job IS transfer learning — understanding when to freeze vs unfreeze layers
- LoRA as a modern form of transfer learning (freeze base, train low-rank adapters)
- Knowing why classifier heads need higher LR than pretrained layers

**CNN Knowledge for Multimodal AI**:
- Vision encoders in CLIP, LLaVA, GPT-4V all use CNNs or ViTs
- Understanding feature extraction for image-to-text pipelines
- Debugging multimodal systems requires understanding both vision and language components

**Regularization in Production**:
- Dropout keeps attention patterns robust in fine-tuned models
- Weight decay (AdamW) prevents catastrophic forgetting during fine-tuning
- Early stopping based on validation loss to find optimal checkpoint

**Architecture Knowledge for System Design**:
- "Design a content moderation system" → understand classification architectures
- "How would you make this model faster?" → quantization, pruning, knowledge distillation
- "Scale this to 100 GPUs" → data/model parallelism decisions

---

## 13. Resources

### Excel Curriculum Links
- Neural Networks: https://www.youtube.com/watch?v=aircAruvnKk (3Blue1Brown)
- Backpropagation: https://www.youtube.com/watch?v=Ilg3gGewQ5U
- CNN Explainer: https://poloclub.github.io/cnn-explainer/
- Stanford CS231n: https://cs231n.stanford.edu/
- PyTorch Tutorials: https://pytorch.org/tutorials/
- TensorFlow Tutorials: https://www.tensorflow.org/tutorials
- ResNet Paper: https://arxiv.org/abs/1512.03385
- EfficientNet Paper: https://arxiv.org/abs/1905.11946
- Transfer Learning Guide: https://cs231n.github.io/transfer-learning/
- Deep Learning Book: https://www.deeplearningbook.org/

### Additional Resources
- fast.ai Practical Deep Learning: https://course.fast.ai/
- Andrej Karpathy Neural Networks: Zero to Hero: https://www.youtube.com/playlist?list=PLAqhIrjkxbuWI23v9cThsA9GvCAUhRvKZ
- d2l.ai - Dive into Deep Learning: https://d2l.ai/
