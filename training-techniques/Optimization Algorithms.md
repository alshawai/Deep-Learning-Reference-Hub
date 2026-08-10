# $\text{Optimization Algorithms in Deep Learning}$

*A comprehensive guide to gradient descent variants and advanced optimization techniques for training efficient deep neural networks*

## Table of Contents
- [Optimization Algorithms in Deep Learning](#textoptimization-algorithms-in-deep-learning)
  - [Table of Contents](#table-of-contents)
  - [Introduction to Optimization](#introduction-to-optimization)
    - [Why Optimization Matters](#why--optimization--matters)
    - [The Optimization Landscape](#the--optimization--landscape)
  - [Gradient Descent Variants](#gradient-descent-variants)
    - [1. Batch Gradient Descent](#1--batch--gradient--descent)
    - [2. Stochastic Gradient Descent (SGD)](#2--stochastic--gradient--descent--sgd)
    - [3. Mini-batch Gradient Descent](#3--mini-batch--gradient--descent)
  - [Exponential Weighted Averages](#exponential-weighted-averages)
    - [Mathematical Foundation](#mathematical--foundation)
    - [Bias Correction](#bias--correction)
  - [Advanced Optimization Algorithms](#advanced-optimization-algorithms)
    - [1. Gradient Descent with Momentum](#1--gradient--descent--with--momentum)
    - [2. RMSprop (Root Mean Square Propagation)](#2--rmsprop--root--mean--square--propagation)
    - [3. Adam (Adaptive Moment Estimation)](#3--adam--adaptive--moment--estimation)
    - [4. Modern Variants and Improvements](#4--modern--variants--and--improvements)
      - [AdamW (Adam with Weight Decay)](#adamw-adam-with-weight-decay)
      - [AMSGrad](#amsgrad)
  - [Learning Rate Scheduling](#learning-rate-scheduling)
    - [Why Learning Rate Scheduling Matters](#why--learning--rate--scheduling--matters)
    - [Common Scheduling Strategies](#common--scheduling--strategies)
      - [1. Step Decay](#1-step-decay)
      - [2. Exponential Decay](#2-exponential-decay)
      - [3. Cosine Annealing](#3-cosine-annealing)
      - [4. Reduce on Plateau](#4-reduce-on-plateau)
  - [Modern Best Practices](#modern-best-practices)
    - [$2024 ~ Recommendations$](#2024--recommendations)
      - [Optimizer Selection](#optimizer-selection)
      - [Hyperparameter Guidelines](#hyperparameter-guidelines)
      - [Training Strategy](#training-strategy)
    - [Framework-Specific Implementations](#framework-specific--implementations)
      - [PyTorch](#pytorch)
      - [TensorFlow/Keras](#tensorflowkeras)
  - [Implementation Examples](#implementation-examples)
    - [Complete Implementations](#complete-implementations)
    - [Utility Functions](#utility-functions)
    - [Complete Training Examples](#complete-training-examples)
  - [Key Takeaways](#key-takeaways)
    - [Quick Reference](#quick-reference)

---

## Introduction to Optimization

### $Why ~ Optimization ~ Matters$

Optimization algorithms are the backbone of deep learning training. They determine how quickly and effectively your model converges to optimal parameters. Poor optimization can lead to:
- **Slow convergence**: Training takes unnecessarily long
- **Poor final performance**: Model gets stuck in suboptimal solutions
- **Training instability**: Loss oscillates or diverges during training
- **Inefficient resource usage**: Wasted computational power and time

### $The ~ Optimization ~ Landscape$

Deep learning optimization involves navigating a high-dimensional, non-convex loss landscape. Numerous challenges arise like:
- **Saddle points**: Points where gradient is zero but not optimal
- **Local minima**: Suboptimal solutions that trap basic gradient descent
- **Vanishing/exploding gradients**: Gradients become too small or too large
- **Ill-conditioned problems**: Different dimensions have vastly different curvatures

**Mathematical Foundation:**  
Given a loss function $J(\theta)$ where $\theta$ represents model parameters, optimization seeks to find:

$$\theta^* = \arg \min_\theta J(\theta)$$  

The iterative update rule for gradient-based optimization is:

$$\theta_{t+1} = \theta_t - \alpha \nabla_\theta J(\theta_t)$$  

where $\alpha$ is the learning rate and $\nabla_\theta J(\theta_t)$ is the gradient w.r.t. $\theta$.

---

## Gradient Descent Variants

### $1. ~ Batch ~ Gradient ~ Descent$

**Classic Approach**: Uses the entire training dataset to compute gradients.

**Algorithm:**
```python
for epoch in range(num_epochs):
    gradient = compute_gradient(X_train, y_train, theta)
    theta -= learning_rate * gradient
```

**Mathematical Formulation:**  

$$\theta_{t+1} = \theta_t - \alpha \frac{1}{m} \sum_{i=1}^{m} \nabla_\theta J(\theta_t, x^{(i)}, y^{(i)})$$  

**Characteristics:**
- ✅ **Stable convergence**: Smooth gradient estimates
- ✅ **Guaranteed convergence**: For convex functions with proper learning rate
- ❌ **Computationally expensive**: Requires full dataset pass per update
- ❌ **Memory intensive**: Must load entire dataset
- ❌ **Slow for large datasets**: Impractical for modern big data scenarios

### $2. ~ Stochastic ~ Gradient ~ Descent ~ (SGD)$

**Single Sample Approach**: Uses one training example at a time.

**Algorithm:**
```python
for epoch in range(num_epochs):
    X_shuffled, y_shuffled = shuffle(
        X_train, y_train
    )  # To ensure randomness in data pattern

    for i in range(m):  # No. of training examples
        gradient = compute_gradient(X_shuffled[i], y_shuffled[i], theta)
        theta -= learning_rate * gradient
```

**Mathematical Formulation:**  

$$\theta_{t+1} = \theta_t - \alpha \nabla_\theta J(\theta_t, x^{(i)}, y^{(i)})$$  

**Characteristics:**
- ✅ **Fast updates**: One gradient computation per update
- ✅ **Memory efficient**: Only one example at a time
- ✅ **Escapes local minima**: Noise helps escape poor solutions
- ✅ **Online learning**: Can adapt to new data in real-time
- ❌ **Noisy convergence**: High variance in gradient estimates
- ❌ **Slow final convergence**: Oscillates around optimum

### $3. ~ Mini-batch ~ Gradient ~ Descent$

**Best of Both Worlds**: Uses small batches of training examples.

**Algorithm:**
```python
for epoch in range(num_epochs):
    mini_batches = create_mini_batches(
        X_train, y_train, batch_size
    )  # To ensure random batches

    for mini_batch in mini_batches:
        X_batch, y_batch = mini_batch
        gradient = compute_gradient(X_batch, y_batch, theta)
        theta -= learning_rate * gradient
```

**Mathematical Formulation:**  

$$\theta_{t+1} = \theta_t - \alpha \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \nabla_\theta J(\theta_t, x^{(i)}, y^{(i)})$$  

where $\mathcal{B}$ is the mini-batch and $|\mathcal{B}|$ is the batch size.

**Characteristics:**
- ✅ **Balanced trade-off**: Combines benefits of both approaches
- ✅ **Vectorization friendly**: Efficient on modern hardware (GPUs)
- ✅ **Reasonable memory usage**: Manageable batch sizes
- ✅ **Stable enough**: Lower variance than pure SGD
- ✅ **Fast enough**: Much faster than batch gradient descent

**Optimal Batch Size Selection:**
- **Small datasets**: 32-64 samples
- **Medium datasets**: 128-256 samples  
- **Large datasets**: 256-512 samples
- **Very large datasets**: 512-1024 samples

**Modern Considerations:**
- Powers of 2 work best with GPU memory architecture
- Larger batches generally give better gradient estimates
- Very large batches may hurt generalization (large batch trap)

---

## Exponential Weighted Averages

### $Mathematical ~ Foundation$

**Core Concept**: Exponential weighted averages (EWA), _or exponentially wieghted moving averages,_ provide a way to compute running averages that give more weight to recent values.

**Formula:**  

$$v_t = \beta v_{t-1} + (1-\beta) \theta_t$$  

where:
- $v_t$ is the exponentially weighted average at time $t$
- $\beta$ is the momentum parameter (typically 0.9-0.999)
- $\theta_t$ is the current value of parameter $\theta$
- $v_0 = 0$ (initialization)

**Intuitive Understanding:**
- $v_t$ approximates the average of the last $\frac{1}{1-\beta}$ values
- $\beta = 0.9$ ≈ average of last 10 values
- $\beta = 0.99$ ≈ average of last 100 values
- $\beta = 0.999$ ≈ average of last 1000 values

### $Bias ~ Correction$

**The Problem**: Early estimates are biased toward zero due to $v_0 = 0$ initialization.

**Solution**: Apply bias correction:  

$$v_t^{corrected} = \frac{v_t}{1 - \beta^t}$$

**Why It Works:**
- At $t=1$: $v_1^{corrected} = \frac{(1-\beta)\theta_1}{1-\beta} = \theta_1$
- As $t \to \infty$: $\beta^t \to 0$, so $v_t^{corrected} \to v_t$  
    ###### $\beta^t \to 0$ as $\beta < 1$

**Implementation:**
```python
def exponential_weighted_average(values, beta=0.9, bias_correction=True):
    v = 0
    corrected_values = []
    
    for t, value in enumerate(values, 1):
        v = beta * v + (1 - beta) * value
        
        if bias_correction:
            v_corrected = v / (1 - beta**t)
            corrected_values.append(v_corrected)
        else:
            corrected_values.append(v)
    
    return corrected_values
```

---

## Advanced Optimization Algorithms

### $1. ~ Gradient ~ Descent ~ with ~ Momentum$

**Core Idea**: Add "momentum" to gradient descent to accelerate convergence and reduce oscillations.  
We update parameter $\theta$ not by $\nabla_\theta J(\theta_t, x^{(i)}, y^{(i)})$, but by $v_t$.

**Algorithm:**
```python
v = np.zeros_like(theta)  # Initialize velocity
for t in range(num_iterations):
    gradient = compute_gradient(X_batch, y_batch, theta)
    
    v = beta * v + (1 - beta) * gradient
    theta -= learning_rate * v
```

**Mathematical Formulation:**  

$$\begin{align}
v_t &= \beta v_{t-1} + (1-\beta) \nabla_\theta J(\theta_t) \\
\theta_{t+1} &= \theta_t - \alpha v_t
\end{align}$$  

**Physical Intuition:**
Think of a ball rolling down a hill:
- **Gradient**: is the current slope direction
- **Momentum**: is the ball's velocity accumulated from previous movements
- **Result**: The ball accelerates in consistent directions, which dampens oscillations

**Benefits:**
- **Faster convergence**: Accelerates in consistent directions
- **Reduced oscillations**: Dampens back-and-forth movements
- **Better local minima escape**: Momentum helps overcome small barriers
- **Robust to noise**: Smooths out noisy gradient estimates

**Hyperparameter Tuning:**
- **$\beta = 0.9$**: Standard choice, works well for most problems
- **$\beta = 0.99$**: More momentum, useful for very noisy gradients
- **$\beta = 0.5$**: Less momentum, useful for quickly changing landscapes

### $2. ~ RMSprop ~ (Root ~ Mean ~ Square ~ Propagation)$

**Core Idea**: Adapt learning rate for each parameter individually based on historical gradient magnitudes.

**Algorithm:**
```python
s = np.zeros_like(theta)
epsilon = 1e-8  # To avoid division by zero

for t in range(num_iterations):
    gradient = compute_gradient(X_batch, y_batch, theta)
    
    s = beta * s + (1 - beta) * gradient**2
    theta -= learning_rate * gradient / (np.sqrt(s) + epsilon)
```

**Mathematical Formulation:**  

$$\begin{align}
s_t &= \beta s_{t-1} + (1-\beta) (\nabla_\theta J(\theta_t))^2 \\
\theta_{t+1} &= \theta_t - \frac{\alpha}{\sqrt{s_t} + \epsilon} \nabla_\theta J(\theta_t)
\end{align}$$  

**Key Insights:**
- **Adaptive learning rates**: Each parameter gets its own effective learning rate
- **Frequent updates get smaller steps**: Parameters with large gradients get smaller learning rates
- **Rare updates get larger steps**: Parameters with small gradients get larger learning rates
- **Automatic scaling**: No need to manually tune learning rate for each parameter

**Benefits:**
- **Handles sparse gradients**: Excellent for problems with sparse features
- **Automatic learning rate adaptation**: Reduces hyperparameter tuning
- **Stable convergence**: Less sensitive to initial learning rate choice
- **Good for non-stationary problems**: Adapts to changing gradient patterns

**Considerations:**
- **Learning rate decay**: Accumulated squared gradients keep growing
- **Hyperparameter**: $\beta = 0.999$ is typical (longer memory for adaptation)

### $3. ~ Adam ~ (Adaptive ~ Moment ~ Estimation)$

**Core Idea**: Combines momentum and RMSprop with bias correction.

**Algorithm:**
```python
m = np.zeros_like(theta)  # First moment (momentum)
v = np.zeros_like(theta)  # Second moment (RMSprop)
epsilon = 1e-8
beta1 = 0.9
beta2 = 0.999

for t in range(1, num_iterations + 1):
    gradient = compute_gradient(X_batch, y_batch, theta)
    m = beta1 * m + (1 - beta1) * gradient
    v = beta2 * v + (1 - beta2) * gradient**2
    
    m_corrected = m / (1 - beta1**t)
    v_corrected = v / (1 - beta2**t)
    
    theta -= learning_rate * m_corrected / (np.sqrt(v_corrected) + epsilon)
```

**Mathematical Formulation:**  
$$\begin{align}
m_t &= \beta_1 m_{t-1} + (1-\beta_1) \nabla_\theta J(\theta_t) \\
v_t &= \beta_2 v_{t-1} + (1-\beta_2) (\nabla_\theta J(\theta_t))^2 \\
\hat{m}_t &= \frac{m_t}{1-\beta_1^t} \\
\hat{v}_t &= \frac{v_t}{1-\beta_2^t} \\
\theta_{t+1} &= \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
\end{align}$$  

**Why Adam Works So Well:**
1. **Momentum component** ($m_t$): Accelerates convergence and reduces oscillations
2. **Adaptive learning rates** ($v_t$): Handles different parameter scales automatically
3. **Bias correction**: Provides accurate estimates especially early in training
4. **Robust default parameters**: Works well across diverse problems

**Benefits:**
- **Best of both worlds**: Combines momentum and adaptive learning rates
- **Excellent default choice**: Works well out-of-the-box for most problems
- **Fast convergence**: Often converges faster than other methods
- **Stable training**: Less sensitive to hyperparameter choices
- **Handles sparse gradients**: Excellent for NLP and sparse data

**Default Hyperparameters:**
- **$\alpha = 0.001$**: Learning rate
- **$\beta_1 = 0.9$**: Momentum parameter
- **$\beta_2 = 0.999$**: RMSprop parameter  
- **$\epsilon = 1e^{-8}$**: Small constant for numerical stability

### $4. ~ Modern ~ Variants ~ and ~ Improvements$

#### AdamW (Adam with Weight Decay)
**Modern Standard**: Fixes weight decay implementation in Adam

**Key Insight**: The regularization effect gets _diluted_ by Original Adam's adaptive learning rates. It applies L2 regularization to the adaptive gradients, which is suboptimal.  

**Algorithm:**
```python
# Standard Adam update
theta = theta - learning_rate * m_corrected / (np.sqrt(v_corrected) + epsilon)
# Added weight decay
theta = theta - learning_rate * weight_decay * theta
```
This way the weight decay becomes consistent, as AdamW applies weight decay directly during parameter update, not to the loss function.   

#### AMSGrad
**Convergence Fix**: Addresses theoretical convergence issues in Adam

**Key Change**: Maintains maximum of past squared gradients instead of exponential average.

---

## Learning Rate Scheduling

### $Why ~ Learning ~ Rate ~ Scheduling ~ Matters$

Learning rate scheduling adapts the learning rate during training to improve convergence:
- **Early training**: Higher learning rate for faster progress
- **Later training**: Lower learning rate for fine-tuning
- **Plateau detection**: Reduce learning rate when progress stalls

### $Common ~ Scheduling ~ Strategies$

#### 1. Step Decay
**Reduce learning rate by a factor at specific epochs:**
```python
def step_decay(epoch, lr):
    if epoch in [30, 60, 90]:
        return lr * 0.1
    return lr
```

#### 2. Exponential Decay
**Gradually reduce learning rate:**
```python
def exponential_decay(epoch, lr):
    return lr * np.exp(-0.1 * epoch)
```

#### 3. Cosine Annealing
**Smooth reduction following cosine curve:**
```python
def cosine_annealing(epoch, lr, T_max):
    return lr * (1 + np.cos(np.pi * epoch / T_max)) / 2
```

#### 4. Reduce on Plateau
**Reduce when validation loss stops improving:**
```python
def reduce_on_plateau(val_loss, lr, patience=10, factor=0.5):
    if no_improvement_for_patience_epochs:  # Core logic needs to be handled.
        return lr * factor
    return lr
```

---

## Modern Best Practices

### $2024 ~ Recommendations$

#### **Optimizer Selection:**
1. **Default choice**: Adam or AdamW for most problems
2. **Computer Vision**: SGD with momentum for final training phase
3. **NLP/Transformers**: AdamW with cosine annealing
4. **Large-scale training**: Adam with gradient clipping

#### **Hyperparameter Guidelines:**
- **Learning rate**: Start with 0.001 for Adam, 0.01 for SGD
- **Batch size**: 32-256 for most problems, larger for very large datasets
- **Momentum**: 0.9 for SGD, 0.9 for Adam's β₁
- **Weight decay**: 0.01-0.1 for regularization

#### **Training Strategy:**
```python
# Modern training loop structure - in PyTorch
optimizer = AdamW(parameters, lr=1e-3, weight_decay=0.01)
scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

for epoch in range(epochs):
    for batch in train_loader:
        loss = model(batch)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )  # Gradient clipping - optional

        optimizer.step()
    scheduler.step()
```

### $Framework-Specific ~ Implementations$

#### **PyTorch:**
```python
import torch.optim as optim

optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999))

optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)

optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
```

#### **TensorFlow/Keras:**
```python
import tensorflow as tf

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

optimizer = tf.keras.optimizers.AdamW(learning_rate=0.001, weight_decay=0.01)

optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
```

---

## Implementation Examples

### Complete Implementations:
> #### **[Mini-batch Gradient Descent](../src/dlhub/optimizers/mini_batch.py)** - Implements efficient mini-batch processing with proper shuffling and batch creation for neural network training.
> #### **[Momentum Optimizer](../src/dlhub/optimizers/momentum.py)** - Complete implementation of gradient descent with momentum, including exponential weighted averages and bias correction.
> #### **[RMSprop Optimizer](../src/dlhub/optimizers/rmsprop.py)** - Adaptive learning rate optimization with squared gradient accumulation and parameter-wise learning rate adjustment.
> #### **[Adam Optimizer](../src/dlhub/optimizers/adam.py)** - Full implementation of the Adam algorithm combining momentum and RMSprop with bias correction.

### Utility Functions:
> #### **[Exponential Weighted Averages](../src/dlhub/optimizers/exponential_weighted_averages.py)** - Standalone implementation of EWA with bias correction for understanding the mathematical foundation.
> #### **[Learning Rate Schedulers](../src/dlhub/optimizers/schedules.py)** - Collection of learning rate scheduling strategies including step decay, exponential decay, and cosine annealing.

### Complete Training Examples:
> #### **[Optimization Comparison](../src/dlhub/optimizers/comparison.py)** - Side-by-side comparison of different optimizers on the same problem, visualizing convergence behavior and final performance.

---

## Key Takeaways

1. **Mini-batch is King**: Mini-batch gradient descent provides the best balance of speed and stability
2. **Adam is the Default**: For most problems, Adam or AdamW should be your first choice
3. **Momentum Matters**: Even simple SGD benefits greatly from momentum
4. **Adaptive Learning Rates**: RMSprop and Adam's adaptive learning rates handle different parameter scales automatically
5. **Bias Correction**: Essential for Adam and other exponential weighted average methods
6. **Learning Rate Scheduling**: Can significantly improve final performance, especially cosine annealing
7. **Modern Practices**: AdamW with weight decay is preferred over Adam with L2 regularization

### **Quick Reference:**
- **Starting a new project**: Use Adam with default parameters
- **Computer vision**: Try SGD with momentum for final training
- **NLP/Transformers**: Use AdamW with cosine annealing
- **Convergence issues**: Check learning rate, add gradient clipping
- **Poor generalization**: Add weight decay, reduce learning rate