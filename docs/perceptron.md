# Perceptron Algorithm

## Table of Contents

1. [The Algorithm](#the-algorithm)
2. [Intuition Behind the Update Rule](#intuition-behind-the-update-rule)
3. [Perceptron Convergence Theorem](#perceptron-convergence-theorem)
4. [Advantages & Disadvantages](#advantages--disadvantages)

## The Algorithm

The Perceptron is an algorithm for learning a binary classifier. The goal is to find a hyperplane that separates the data linearly. It predicts labels $\hat{y} \in \{-1, +1\}$ given an input $x \in \mathbb{R}^n$.

### **Algorithm 1: Perceptron Algorithm**

**Input:** Sequence of training examples $(x_1, y_1), (x_2, y_2), \ldots$, where $x_i \in \mathbb{R}^n$, $y_i \in \{-1, +1\}$

**Output:** Final weight vector $w \in \mathbb{R}^n$

```text
1. Initialize:      w₁ = 0 ∈ ℝⁿ
2. For t = 1, 2, ...
3.     Sample (xᵢ, yᵢ)
4.     Predict:      ŷ = sign(wₜᵀ xᵢ)
5.     If ŷ ≠ yᵢ then
6.         wₜ₊₁ = wₜ + yᵢ xᵢ
7.     Else
8.         wₜ₊₁ = wₜ
```

## Intuition Behind the Update Rule

### Why is the update rule:

$$
w_{t+1} = w_t + y_i x_i\ ?
$$

We can analyze this by multiplying the transpose of the weight update with the feature vector $x_i$:

$$
\begin{aligned}
w_{t+1}^\top x_i &= (w_t + y_i x_i)^\top x_i \\
                 &= w_t^\top x_i + y_i x_i^\top x_i
\end{aligned}
$$

Let’s consider two cases:

### Case 1: $y_i = +1$ (positive example)

If the prediction was wrong, then $w_t^\top x_i < 0$, and we have:

$$
w_{t+1}^\top x_i = \underbrace{w_t^\top x_i}_{< 0} + \underbrace{x_i^\top x_i}_{\|x_i\|^2 > 0}
$$

This moves the dot product closer to being positive, which corrects the classification.

### Case 2: $y_i = -1$ (negative example)

If the prediction was wrong, then $w_t^\top x_i > 0$, and we have:

$$
w_{t+1}^\top x_i = \underbrace{w_t^\top x_i}_{> 0} - \underbrace{x_i^\top x_i}_{\|x_i\|^2 > 0}
$$

This moves the dot product closer to being negative, again correcting the classification.

## Perceptron Convergence Theorem

### **Theorem (Perceptron Mistake Bound)**

Assume:

* The data is linearly separable if there exists a unit vector $w^* \in \mathbb{R}^n$ and margin $\gamma > 0$ such that

  $$
  \forall i,\quad y_i (w^{*T} x_i) \geq \gamma
  $$
* Each example is bounded: $\|x_i\| \leq R$

Then, the Perceptron algorithm makes at most $\frac{R^2}{\gamma^2}$ mistakes.

### **Proof:**

When mistake $k$ occurs at example $t$, we update:

$$
w_{k+1} = w_k + y_t x_t
$$

Multiply both sides by $w^*$:

$$
w_{k+1}^\top w^* = w_k^\top w^* + y_t x_t^\top w^* \geq w_k^\top w^* + \gamma
$$

By induction (starting from $w_1 = 0$):

$$
w_{k+1}^\top w^* \geq k \gamma \tag{1}
$$

Now compute the squared norm of $w_{k+1}$:

$$
\begin{align*}
\|w_{k+1}\|^2 &= \|w_k + y_t x_t\|^2 \\
              &= \|w_k\|^2 + 2 y_t w_k^\top x_t + \|x_t\|^2 \\
              &\leq \|w_k\|^2 + R^2 \\
              &\leq k R^2 \tag{2}
\end{align*}
$$

Using Cauchy-Schwarz and equations (1) and (2):

$$
\begin{aligned}
\|w_{k+1}\| \cdot \|w^*\| &\geq w_{k+1}^\top w^* \\
\Rightarrow \|w_{k+1}\| &\geq k \gamma \quad \text{(since \( \|w^*\| = 1 \))} \\
\Rightarrow (k\gamma)^2 &\leq k R^2 \Rightarrow k \leq \frac{R^2}{\gamma^2}
\end{aligned}
$$

Hence the Perceptron algorithm makes at most $\frac{R^2}{\gamma^2}$ mistakes..

## Advantages & Disadvantages

### Advantages

1. Cheap and easy to implement.
2. Online algorithm: processes one example at a time. It can scale to large datasets.
3. Can use any features easily.

### Disadvantages

1. All classifiers that separate the data are equivalent.
2. Convergence depends on the margin $\gamma$. If classes are not well separated, slow convergence.