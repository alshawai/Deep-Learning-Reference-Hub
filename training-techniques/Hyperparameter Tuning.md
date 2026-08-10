# $\text{Hyperparameter Tuning in Deep Learning}$

*A comprehensive guide to modern hyperparameter optimization strategies, from traditional methods to cutting-edge automated approaches*

## Table of Contents
- [Hyperparameter Tuning in Deep Learning](#texthyperparameter-tuning-in-deep-learning)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
    - [Why Hyperparameter Tuning Matters](#why--hyperparameter--tuning--matters)
  - [Understanding Hyperparameters](#understanding-hyperparameters)
    - [Classification by Impact Level](#classification--by--impact--level)
      - [Tier 1: Critical Hyperparameters (High Impact)](#tier-1-critical-hyperparameters-high-impact)
      - [Tier 2: Important Hyperparameters (Medium Impact)](#tier-2-important-hyperparameters-medium-impact)
      - [Tier 3: Fine-tuning Hyperparameters (Lower Impact)](#tier-3-fine-tuning-hyperparameters-lower-impact)
    - [Modern Hyperparameter Categories](#modern--hyperparameter--categories)
      - [Transformer-Specific Hyperparameters](#transformer-specific-hyperparameters)
  - [Traditional Tuning Strategies](#traditional-tuning-strategies)
    - [1. Manual Search](#1--manual--search)
    - [2. Grid Search](#2--grid--search)
    - [3. Random Search](#3--random--search)
  - [Modern Optimization Methods](#modern-optimization-methods)
    - [1. Bayesian Optimization](#1--bayesian--optimization)
    - [2. Multi-Fidelity Optimization](#2--multi-fidelity--optimization)
    - [3. Population-Based Training (PBT)](#3--population-based--training--pbt)
    - [4. Automated Machine Learning (AutoML)](#4--automated--machine--learning--automl)
  - [Practical Guidelines and Best Practices](#practical-guidelines-and-best-practices)
    - [Recent Best Practices](#recent--best--practices)
      - [Starting Strategy (Modern Recommended Approach)](#starting-strategy-modern-recommended-approach)
      - [Resource-Aware Tuning](#resource-aware-tuning)
      - [Domain-Specific Guidelines](#domain-specific-guidelines)
    - [Common Pitfalls and Solutions](#common--pitfalls--and--solutions)
      - [Pitfall 1: Overfitting to Validation Set](#pitfall-1-overfitting-to-validation-set)
      - [Pitfall 2: Ignoring Statistical Significance](#pitfall-2-ignoring-statistical-significance)
      - [Pitfall 3: Poor Search Space Design](#pitfall-3-poor-search-space-design)
      - [Pitfall 4: Computational Inefficiency](#pitfall-4-computational-inefficiency)
  - [Advanced Topics](#advanced-topics)
    - [Multi-Objective Optimization](#multi-objective--optimization)
    - [Transfer Learning for Hyperparameters](#transfer--learning--for--hyperparameters)
    - [Hyperparameter Importance Analysis](#hyperparameter--importance--analysis)
  - [Implementation Examples](#implementation-examples)
    - [Individual Techniques](#individual-techniques)
    - [Advanced Implementations](#advanced-implementations)
    - [Complete Frameworks](#complete-frameworks)
  - [Key Takeaways](#key-takeaways)

---

## Introduction

Hyperparameter tuning remains one of the most critical yet challenging aspects of deep learning. Deep learning has achieved tremendous success in the past decade, sparking a revolution in artificial intelligence. However, the modern practice of deep learning remains largely an art form, requiring a delicate combination of guesswork and careful hyperparameter tuning.

Unlike model parameters (weights and biases) that are learned during training, hyperparameters are configuration settings that must be specified before training begins. The choice of hyperparameters can dramatically affect model performance, training stability, and convergence speed.

### $Why ~ Hyperparameter ~ Tuning ~ Matters$

**Impact on Model Performance:**
- Learning rate affects convergence speed and final accuracy
- Architecture choices determine model capacity and expressiveness  
- Regularization parameters control the bias-variance tradeoff
- Batch size influences gradient estimates and memory requirements

**With the rise of large language models and transformer architectures**, hyperparameter tuning has become even more crucial due to:
- **Computational costs**: Training large models is expensive, making efficient tuning essential
- **Emergent behaviors**: Complex interactions between hyperparameters in deep networks
- **Scale sensitivity**: Hyperparameters that work at small scales may fail at large scales

---

## Understanding Hyperparameters

### $Classification ~ by ~ Impact ~ Level$

#### **Tier 1: Critical Hyperparameters** (High Impact)
These have the most significant effect on model performance and should be tuned first:

1. **Learning Rate (α)**
   - **Range**: Typically $10^{-5}$ to $10^{-1}$
   - **Architecture dependency**: Transformers often need lower rates than CNNs
Using a _learning rate scheduler_ and _warmup strategies_ can greatly impact the training speed and accuracy of the models.
> For more details about **learning rate schedulers**, read the related topic in [Optimization Algorithms](./Optimization%20Algorithms.md)
> _You can find learning rate schedulers implementation using numpy in [Examples](#implementation-examples)_

2. **Architecture Parameters**
   - **Depth**: Number of layers
   - **Width**: Hidden units per layer
   - **Attention heads**: For transformer models

3. **Batch Size**
   - **Range**: Powers of 2, typically 16-512 for most applications
   - **Modern consideration**: Large batch training (>1024) requires special techniques

#### **Tier 2: Important Hyperparameters** (Medium Impact)

1. **Optimization Parameters**
   - **Beta values** for Adam: $\beta_1 \in [0.9, 0.99]$, $\beta_2 \in [0.99, 0.999]$
   - **Weight decay**: $\lambda \in [10^{-6}, 10^{-2}]$

2. **Regularization**
   - **Dropout rate**: $p \in [0.1, 0.5]$
   - **Label smoothing**: $\epsilon \in [0.05, 0.2]$

#### **Tier 3: Fine-tuning Hyperparameters** (Lower Impact)

1. **Numerical Stability**
   - **Epsilon values**: For batch norm, layer norm
   - **Gradient clipping**: Threshold values

2. **Training Dynamics**
   - **Warmup steps**: For learning rate scheduling
   - **Decay factors**: For exponential decay

### $Modern ~ Hyperparameter ~ Categories$

#### **Transformer-Specific Hyperparameters**

1. **Attention Mechanism**
   - **Number of attention heads**: Typically 8, 12, or 16
   - **Head dimension**: Usually 64 or 128
   - **Attention dropout**: Separate from regular dropout

2. **Position Encoding**
   - **Maximum sequence length**: Context window size
   - **Position embedding type**: Learned vs. sinusoidal

3. **Layer Configuration**
   - **Feed-forward ratio**: Typically 4x the model dimension
   - **Pre/post layer normalization**: Architectural choice

---

## Traditional Tuning Strategies

### $1. ~ Manual ~ Search$

**Process:**
1. Start with known good defaults
2. Adjust one hyperparameter at a time
3. Use domain knowledge and intuition

**Advantages:**
- Educational value for understanding model behavior
- Can incorporate domain expertise
- Good for initial exploration

**Disadvantages:**
- Time-intensive and doesn't scale
- May miss optimal combinations
- Prone to human bias and local optima

### $2. ~ Grid ~ Search$

**Mathematical Formulation:**
For hyperparameters $\theta_1, \theta_2, ..., \theta_k$ with discrete sets of values:
$$\Theta = \{\theta_1^{(1)}, \theta_1^{(2)}, ...\} \times \{\theta_2^{(1)}, \theta_2^{(2)}, ...\} \times ... \times \{\theta_k^{(1)}, \theta_k^{(2)}, ...\}$$

**Example Implementation:**
```python
learning_rates = [0.1, 0.01, 0.001]
batch_sizes = [32, 64, 128]
hidden_units = [128, 256, 512]

# Total combinations: 3 × 3 × 3 = 27 experiments
for lr in learning_rates:
    for bs in batch_sizes:
        for hu in hidden_units:
            model = train_model(lr=lr, batch_size=bs, hidden_units=hu)
            evaluate_model(model)
```

**Advantages:**
- Systematic and reproducible
- Guarantees finding the best combination within the grid
- Parallelizable across multiple machines

**Disadvantages:**  
- **Curse of dimensionality**: Exponential growth in search space
- **Inefficient sampling**: Wastes computation on poor regions
- **Discrete limitation**: May miss optimal values between grid points

### $3. ~ Random ~ Search$

**Mathematical Foundation:**
Sample hyperparameters from probability distributions:
$$\theta_i \sim P_i(\theta_i) \quad \text{for } i = 1, 2, ..., k$$

**Key Insight (Bergstra & Bengio, 2012):**
Random search is more efficient than grid search because:
- Many hyperparameters have low effective dimensionality
- Random search explores more unique values per dimension
- Better coverage of the search space with fewer evaluations

**Modern Implementation:**
```python
import numpy as np
from scipy.stats import loguniform, uniform


def sample_hyperparameters():
    return {
        "learning_rate": loguniform.rvs(1e-5, 1e-1),
        "batch_size": np.random.choice([16, 32, 64, 128, 256]),
        "hidden_units": np.random.randint(64, 512),
        "dropout_rate": uniform.rvs(0.1, 0.4),
        "weight_decay": loguniform.rvs(1e-6, 1e-2),
    }


# Sample N random configurations
configurations = [sample_hyperparameters() for _ in range(100)]
```

**Distribution Guidelines:**
- **Learning rate**: Log-uniform distribution
- **Regularization parameters**: Log-uniform distribution  
- **Discrete choices**: Uniform sampling from valid options
- **Architecture parameters**: Often uniform or integer distributions

> _You can find random search implementation using numpy in [Examples](#implementation-examples)_
---

## Modern Optimization Methods

### $1. ~ Bayesian ~ Optimization$

**Core Principle:**
Build a probabilistic model of the objective function $f(\theta)$ and use it to guide the search toward promising regions.

**Mathematical Framework:**
1. **Surrogate Model**: Use Gaussian Process (GP) to model $f(\theta)$
   $$f(\theta) \sim \mathcal{GP}(\mu(\theta), k(\theta, \theta'))$$

2. **Acquisition Function**: Balance exploration vs. exploitation
   - **Expected Improvement (EI)**:
     $$EI(\theta) = \mathbb{E}[\max(0, f(\theta) - f^*)]$$
   - **Upper Confidence Bound (UCB)**:
     $$UCB(\theta) = \mu(\theta) + \kappa \sigma(\theta)$$

3. **Optimization Loop**:
   ```
   For iteration t = 1, 2, ...:
     1. Fit GP to observed data {(θᵢ, f(θᵢ))}ᵢ₌₁ᵗ⁻¹
     2. Find θₜ = argmax acquisition_function(θ)
     3. Evaluate f(θₜ) and add to dataset
   ```

**Modern Tools:**
- **Optuna**: Popular Python library with advanced features
- **Hyperopt**: TPE (Tree-structured Parzen Estimator) algorithm
- **Google Vizier**: Large-scale Bayesian optimization
- **Weights & Biases Sweeps**: Integrated with experiment tracking

**Advantages:**
- Sample-efficient, especially for expensive evaluations
- Principled uncertainty quantification
- Can handle mixed variable types (continuous, discrete, categorical)

**Limitations:**
- Computational overhead grows with number of evaluations
- Assumes smoothness in the objective function
- May struggle with very high-dimensional spaces (>20 dimensions)

> _You can find Bayesian optimization implementation using numpy in [Examples](#implementation-examples)_


### $2. ~ Multi-Fidelity ~ Optimization$

**Concept:**
Use cheaper approximations (lower fidelity) to guide the search, then evaluate promising candidates at full fidelity.

**Fidelity Dimensions:**
1. **Training time**: Early stopping vs. full training
2. **Data size**: Subset vs. full dataset  
3. **Model size**: Smaller models vs. target architecture
4. **Resolution**: Lower vs. higher input resolution (for vision tasks)

**Successive Halving Algorithm:**
```python
def successive_halving(configurations, budget):
    active_configs = configurations.copy()
    budget_per_config = budget // len(configurations)

    while len(active_configs) > 1:
        # Train all active configurations with current budget
        results = []
        for config in active_configs:
            score = train_model(config, budget=budget_per_config)
            results.append((config, score))

        # Keep top half, double the budget
        results.sort(key=lambda x: x[1], reverse=True)
        active_configs = [config for config, _ in results[: len(results) // 2]]
        budget_per_config *= 2

    return active_configs[0]  # Best configuration
```

**ASHA (Asynchronous Successive Halving):**
- Extends successive halving to asynchronous/parallel setting
- Used in many modern hyperparameter tuning frameworks

> _You can find multi-fidelity optimization implementation using numpy in [Examples](#implementation-examples)_

### $3. ~ Population-Based ~ Training ~ (PBT)$

**Innovative Approach (DeepMind, 2017):**
Simultaneously train multiple models with different hyperparameters and periodically update hyperparameters based on population performance.

**Algorithm:**
1. **Initialize**: Population of models with different hyperparameters
2. **Train**: Each model trains independently for a period
3. **Evaluate**: Assess performance of all models
4. **Exploit**: Replace poor-performing models with copies of better ones
5. **Explore**: Perturb hyperparameters of the copied models
6. **Repeat**: Continue training with updated population

**Key Benefits:**
- Online hyperparameter adaptation during training
- Can discover time-varying optimal hyperparameters
- Particularly effective for long training runs

**Modern Applications:**
- Large language model training
- Reinforcement learning
- Neural architecture search

> _You can find population-based training implementation using numpy in [Examples](#implementation-examples)_

### $4. ~ Automated ~ Machine ~ Learning ~ (AutoML)$

**Neural Architecture Search (NAS):**
Automated discovery of neural network architectures using:
- **Reinforcement learning**: Controller network proposes architectures
- **Evolutionary algorithms**: Mutation and selection of architectures  
- **Differentiable NAS**: Gradient-based optimization over architecture space

**Progressive Approaches:**
- **Progressive search**: Start simple, gradually increase complexity
- **Weight sharing**: Amortize training cost across architecture candidates
- **One-shot models**: Train supernet containing all possible architectures

---

## Practical Guidelines and Best Practices

### $Recent ~ Best ~ Practices$

#### **Starting Strategy (Modern Recommended Approach)**

1. **Baseline First** (Day 1):
   ```python
   # Start with known good defaults
   config = {
       "learning_rate": 3e-4,
       "batch_size": 32,
       "weight_decay": 1e-4,
       "dropout": 0.1,
       "warmup_steps": 1000,
   }
   ```

2. **Learning Rate Tuning** (Days 2-3):
   - Use learning rate finder/range test
   - Start with range $[10^{-5}, 10^{-1}]$
   - Use cosine annealing or linear warmup + decay

> _You can find learning rate finder implementation using numpy in [Examples](#implementation-examples)_

3. **Architecture Scaling** (Days 4-5):
   - Scale model size based on data complexity
   - Use established scaling laws when available

4. **Advanced Optimization** (Week 2+):
   - Bayesian optimization for fine-tuning
   - Multi-objective optimization if needed

#### **Resource-Aware Tuning**

**For Limited Compute:**
1. Start with random search (20-50 trials)
2. Use early stopping aggressively  
3. Focus on most impactful hyperparameters (Tier 1)
4. Use small models for initial exploration

**For Abundant Compute:**
1. Comprehensive Bayesian optimization
2. Multi-fidelity approaches
3. Population-based training for long runs
4. Neural architecture search

#### **Domain-Specific Guidelines**

**Computer Vision:**
```python
vision_defaults = {
    "learning_rate": 1e-3,  # Higher for CNNs
    "weight_decay": 1e-4,  # Important for generalization
    "batch_size": 64,  # Balance memory/gradient quality
    "augmentation_strength": 0.2,  # Data augmentation intensity
    "mixup_alpha": 0.2,  # Modern regularization
}
```

**Natural Language Processing:**
```python
nlp_defaults = {
    "learning_rate": 5e-5,  # Lower for pre-trained models
    "warmup_ratio": 0.1,  # Warmup proportion
    "max_grad_norm": 1.0,  # Gradient clipping
    "label_smoothing": 0.1,  # For classification
    "attention_dropout": 0.1,  # Separate attention dropout
}
```

**Large Language Models:**
```python
llm_defaults = {
    "learning_rate": 1e-4,  # Very careful tuning needed
    "batch_size": 2048,  # Large batch training
    "gradient_accumulation": 8,  # Simulate larger batches
    "beta2": 0.95,  # Lower beta2 often better
    "weight_decay": 0.1,  # Higher for large models
}
```

### $Common ~ Pitfalls ~ and ~ Solutions$

#### **Pitfall 1: Overfitting to Validation Set**
**Problem**: Repeatedly evaluating on the same validation set leads to overfitting.

**Solutions:**
- Use multiple validation splits (cross-validation)
- Hold out a separate test set for final evaluation
- Use statistical significance testing
- Limit the number of hyperparameter evaluations

#### **Pitfall 2: Ignoring Statistical Significance**
**Problem**: Choosing hyperparameters based on single runs with random variation.

**Solutions:**
```python
# Run multiple seeds and test significance
import scipy.stats as stats


def compare_configurations(config_a_scores, config_b_scores):
    """Compare two configurations with statistical testing."""
    statistic, p_value = stats.ttest_ind(config_a_scores, config_b_scores)

    if p_value < 0.05:
        return "Statistically significant difference"
    else:
        return "No significant difference"


# Example usage
config_a_scores = [0.92, 0.91, 0.93, 0.90, 0.92]  # 5 different seeds
config_b_scores = [0.89, 0.90, 0.88, 0.91, 0.89]

result = compare_configurations(config_a_scores, config_b_scores)
```

#### **Pitfall 3: Poor Search Space Design**
**Problem**: Inappropriate ranges or distributions for hyperparameters.

**Solutions:**
- Use log-scale for multiplicative parameters (learning rate, weight decay)
- Use appropriate priors based on domain knowledge
- Start with wide ranges, then narrow down

#### **Pitfall 4: Computational Inefficiency**
**Problem**: Wasting compute on clearly poor configurations.

**Solutions:**
- Use early stopping based on learning curves
- Implement multi-fidelity optimization
- Parallel evaluation across multiple GPUs/machines

---

## Advanced Topics

### $Multi-Objective ~ Optimization$

In real applications, we often care about multiple objectives:
- **Accuracy vs. Latency**: Model performance vs. inference speed
- **Accuracy vs. Model Size**: Performance vs. memory requirements  
- **Accuracy vs. Fairness**: Performance vs. bias metrics

**Pareto Optimization:**
Find the set of non-dominated solutions (Pareto frontier):
```python
def is_pareto_optimal(scores, maximize=[True, False]):
    """
    Find Pareto optimal solutions.
    scores: array of shape (n_points, n_objectives)
    maximize: list indicating whether to maximize each objective
    """
    is_efficient = np.ones(scores.shape[0], dtype=bool)
    
    for i, score in enumerate(scores):
        if is_efficient[i]:
            # Compare with all other points
            for j, other_score in enumerate(scores):
                if i != j and is_efficient[j]:
                    dominates = True
                    for k, (maximize_k) in enumerate(maximize):
                        if maximize_k:
                            if score[k] < other_score[k]:
                                dominates = False
                                break
                        else:
                            if score[k] > other_score[k]:
                                dominates = False
                                break
                    
                    if dominates:
                        is_efficient[j] = False
    
    return is_efficient
```

### $Transfer ~ Learning ~ for ~ Hyperparameters$

**Concept**: Use knowledge from previous hyperparameter tuning experiments to initialize new searches.

**Approaches:**
1. **Meta-learning**: Learn a mapping from dataset/architecture characteristics to good hyperparameters
2. **Warm-starting**: Initialize Bayesian optimization with results from similar problems
3. **Few-shot hyperparameter optimization**: Quickly adapt hyperparameters for new tasks

### $Hyperparameter ~ Importance ~ Analysis$

**Functional ANOVA Decomposition:**
Decompose the objective function to understand hyperparameter importance:
$$f(\theta) = f_0 + \sum_i f_i(\theta_i) + \sum_{i<j} f_{ij}(\theta_i, \theta_j) + ...$$

**Practical Tool:**
```python
from sklearn.ensemble import RandomForestRegressor
import numpy as np


def analyze_hyperparameter_importance(X_params, y_scores):
    """
    Analyze which hyperparameters are most important.
    X_params: array of hyperparameter configurations
    y_scores: corresponding performance scores
    """
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_params, y_scores)

    importances = rf.feature_importances_
    param_names = ["learning_rate", "batch_size", "hidden_units", ...]

    # Sort by importance
    sorted_idx = np.argsort(importances)[::-1]

    print("Hyperparameter Importance Ranking:")
    for i, idx in enumerate(sorted_idx):
        print(f"{i + 1}. {param_names[idx]}: {importances[idx]:.3f}")

    return importances
```

---

## Implementation Examples

### Individual Techniques:
> #### **[Learning Rate Scheduler](../src/dlhub/optimizers/schedules.py)** - Collection of learning rate scheduling strategies including step decay, exponential decay, and cosine annealing.
>
> #### **[Bayesian Optimization](../code-examples/numpy/hyperparameter_tuning/bayesian_optimization.py)** - Implementation using Gaussian Process surrogate models with Expected Improvement acquisition function for efficient hyperparameter search.
> 
> #### **[Random Search](../src/dlhub/tuning/random_search.py)** - Comprehensive random search implementation with proper probability distributions and parallel evaluation support.
> 
> #### **[Learning Rate Finder](../code-examples/numpy/learning_rate_finder.py)** - Automated learning rate range testing to find optimal learning rate ranges before full training.

### Advanced Implementations:
> #### **[Multi-Fidelity Optimization](../code-examples/numpy/hyperparameter_tuning/multifidelity_optimization.py)** - ASHA (Asynchronous Successive Halving) implementation for efficient resource allocation across hyperparameter candidates.
> 
> #### **[Population-Based Training](../code-examples/numpy/hyperparameter_tuning/population_based_training.py)** - Complete PBT implementation with online hyperparameter adaptation during training.

### Complete Frameworks:
> #### **[Modern Hyperparameter Tuning Framework](../code-examples/numpy/hyperparameter_tuning/complete_tuning_framework.py)** - Production-ready framework integrating multiple optimization strategies with experiment tracking and statistical analysis.

---

## Key Takeaways

1. **Prioritize Impact**: Focus on high-impact hyperparameters (learning rate, architecture) before fine-tuning details

2. **Use Modern Methods**: Bayesian optimization and multi-fidelity approaches are more efficient than grid/random search for expensive evaluations

3. **Statistical Rigor**: Always use multiple random seeds and statistical testing when comparing configurations

4. **Resource Awareness**: Choose tuning strategy based on available computational resources

5. **Domain Knowledge**: Incorporate domain-specific best practices and scaling laws

6. **Automation**: Use modern AutoML tools when appropriate, but understand the underlying principles

7. **Continuous Learning**: The field evolves rapidly; stay updated with latest research and tools

**Final Insight**: Recent effective hyperparameter tuning combines principled optimization methods with domain expertise and computational efficiency. The goal is not just finding good hyperparameters, but doing so efficiently and reliably at scale.