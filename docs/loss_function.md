# Loss Functions

## Table of Contents

1. [Introduction](#introduction)
2. [Mean Squared Error (MSE)](#mean-squared-error-mse)
3. [Mean Absolute Error (MAE)](#mean-absolute-error-mae)
4. [Huber Loss](#huber-loss)
5. [Binary Cross-Entropy Loss](#binary-cross-entropy-loss)
6. [Categorical Cross-Entropy Loss](#categorical-cross-entropy-loss)
7. [Kullback-Leibler Divergence (KL Divergence)](#kullback-leibler-divergence-kl-divergence)
8. [Hinge Loss](#hinge-loss)
9. [Contrastive Loss](#contrastive-loss)
10. [Triplet Loss](#triplet-loss)
11. [Conclusion](#conclusion)

## Introduction

Loss functions (also called **cost functions** or **objective functions**) quantify the difference between predicted outputs and true targets. During training, neural networks use loss functions to guide weight updates using optimization algorithms like Gradient Descent. A good choice of loss function ensures meaningful learning and faster convergence.

Given:

* Predictions: $\hat{y}$
* True values: $y$
* Loss: $L(y, \hat{y})$

The loss function is minimized during training.

## Mean Squared Error (MSE)

Used in regression tasks.

### Definition

$$
L(y, \hat{y}) = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2
$$

### Properties

* Penalizes large errors more heavily than small errors.
* Sensitive to outliers.
* Smooth and differentiable.

### Derivative

$$
\frac{\partial L}{\partial \hat{y}_i} = -2(y_i - \hat{y}_i)
$$

## Mean Absolute Error (MAE)

Also for regression tasks.

### Definition

$$
L(y, \hat{y}) = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|
$$

### Properties

* More robust to outliers than MSE.
* Not differentiable at $y = \hat{y}$ (but subgradients are used).
* Slower convergence compared to MSE.

### Derivative (subgradient)

$$
\frac{\partial L}{\partial \hat{y}_i} =
\begin{cases}
-1 & \text{if } y_i - \hat{y}_i > 0 \\
1 & \text{if } y_i - \hat{y}_i < 0
\end{cases}
$$

## Huber Loss

Combines MSE and MAE to be robust and smooth.

### Definition

$$
L_\delta(y, \hat{y}) =
\begin{cases}
\frac{1}{2}(y - \hat{y})^2 & \text{if } |y - \hat{y}| \leq \delta \\
\delta \left( |y - \hat{y}| - \frac{1}{2}\delta \right) & \text{otherwise}
\end{cases}
$$

### Properties

* Quadratic for small errors, linear for large errors.
* Reduces sensitivity to outliers.

## Binary Cross-Entropy Loss

Used in binary classification.

### Definition

$$
L(y, \hat{y}) = -[y \log(\hat{y}) + (1 - y)\log(1 - \hat{y})]
$$

Where $y \in {0, 1}$ and $\hat{y} \in (0, 1)$.

### Properties

* Penalizes confident incorrect predictions.
* Assumes outputs are probabilities (usually after Sigmoid).

### Derivative

$$
\frac{\partial L}{\partial \hat{y}} = -\frac{y}{\hat{y}} + \frac{1 - y}{1 - \hat{y}}
$$

## Categorical Cross-Entropy Loss

Used in multi-class classification with one-hot encoded targets.

### Definition

Given:

* True label vector $y \in \{0, 1\}^C$ (one-hot encoded)
* Predicted probabilities $\hat{y} = \text{softmax}(z)$

$$
L(y, \hat{y}) = -\sum_{i=1}^C y_i \log(\hat{y}_i)
$$

Where:

* $C$ = number of classes
* $y_i = 1$ for the correct class, \$0\$ otherwise

### With Softmax Output:

Let:

$$
\hat{y}_i = \frac{e^{z_i}}{\sum_{j=1}^C e^{z_j}} \quad \text{(Softmax)}
$$

### Properties

* Used with Softmax outputs.
* Measures log loss over multiple categories.


## Derivative of Loss w.r.t. Input $z_i$

We want to compute the derivative of the loss with respect to the logits $z_k$, where $\hat{y}_k = \text{softmax}(z_k)$.

### Step-by-Step:

$$
\frac{\partial L}{\partial z_k} = \sum_{i=1}^C \frac{\partial L}{\partial \hat{y}_i} \cdot \frac{\partial \hat{y}_i}{\partial z_k}
$$

We already know:

$$
\frac{\partial L}{\partial \hat{y}_i} = -\frac{y_i}{\hat{y}_i}
$$

Now, for softmax:

$$
\frac{\partial \hat{y}_i}{\partial z_k} =
\begin{cases}
\hat{y}_i(1 - \hat{y}_i) & \text{if } i = k \\
-\hat{y}_i \hat{y}_k & \text{if } i \ne k
\end{cases}
$$

Putting this together:

$$
\frac{\partial L}{\partial z_k} = \hat{y}_k - y_k
$$


This is more common and efficient: when using categorical cross-entropy with softmax, the gradient simplifies greatly, which is why they are often combined in frameworks like TensorFlow (`SoftmaxCrossEntropyWithLogits`) and PyTorch (`CrossEntropyLoss`).


## Kullback-Leibler Divergence (KL Divergence)

Measures how one probability distribution diverges from a second expected distribution.

### Definition

$$
D_{KL}(P \parallel Q) = \sum_i P(i) \log \frac{P(i)}{Q(i)}
$$

Where:

* $P$: true distribution
* $Q$: predicted distribution

### Properties

* Asymmetric: $D\_{KL}(P \parallel Q) \neq D\_{KL}(Q \parallel P)$
* Used in VAEs (Variational Autoencoders), NLP models

## Hinge Loss

Used in SVMs and "maximum-margin" classifiers.

### Definition

$$
L(y, \hat{y}) = \max(0, 1 - y \cdot \hat{y})
$$

Where:

* $y \in {-1, +1}$

### Properties

* Encourages correct classification with a margin.
* Only penalizes incorrect or borderline predictions.


## Contrastive Loss

Used in **Siamese Networks** (e.g., face recognition).

### Definition

$$
L = (1 - y) \cdot \frac{1}{2} D^2 + y \cdot \frac{1}{2} \max(0, m - D)^2
$$

Where:

* $D$: Euclidean distance between feature embeddings
* $y = 0$ for similar pairs, $1$ for dissimilar
* $m$: margin


## Triplet Loss

Used in ranking tasks (e.g., face verification).

### Definition

$$
L = \max(0, d(a, p) - d(a, n) + \alpha)
$$

Where:

* $a$ = anchor
* $p$ = positive example
* $n$ = negative example
* $\alpha$ = margin

### Properties

* Encourages anchor-positive pairs to be closer than anchor-negative pairs by a margin $\alpha$.

## Conclusion

Loss functions guide learning by quantifying prediction errors. Choosing the right loss function depends on the problem type:

| Task Type                  | Common Loss Functions                           |
| -------------------------- | ----------------------------------------------- |
| Regression                 | MSE, MAE, Huber                                 |
| Binary Classification      | Binary Cross-Entropy                            |
| Multi-class Classification | Categorical Cross-Entropy, Sparse Cross-Entropy |
| Embedding / Similarity     | Contrastive, Triplet                            |
| Structured / Probabilistic | KL Divergence, Hinge                            |

Proper loss function choice can significantly improve convergence, performance, and generalization.