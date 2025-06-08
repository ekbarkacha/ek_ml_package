# Gaussian Discriminant Analysis (GDA) — Multiclass

## Table of Contents

1. [Introduction](#introduction)
2. [Generative Assumptions](#generative-assumptions)
3. [Model Derivation](#model-derivation)
4. [Linear Discriminant Analysis (LDA)](#linear-discriminant-analysis-lda)
5. [Quadratic Discriminant Analysis (QDA)](#quadratic-discriminant-analysis-qda)
6. [Comparison: LDA vs QDA](#comparison-lda-vs-qda)
7. [Conclusion](#conclusion)


## Introduction

GDA is a generative classifier as it models the joint distribution $P(x, y)$, and uses Bayes' Rule to derive $P(y|x)$ for classification.

GDA assumes that the feature distribution conditioned on the class is Gaussian:

$$
P(x \mid y = k) \sim \mathcal{N}(\mu_k, \Sigma_k)
$$

This generalizes easily to the multiclass setting, where:

* $x \in \mathbb{R}^d$ is the feature vector
* $y \in \{0, 1, ..., K-1\}$ is the class label
* $\phi_k = P(y = k)$ is the prior for class $k$
* $\mu_k \in \mathbb{R}^d$, $\Sigma_k \in \mathbb{R}^{d \times d}$

## Generative Assumptions

Let the training data be:

$$
\mathcal{D} = \left\{ (x_1, y_1), (x_2, y_2), ..., (x_n, y_n) \right\}
$$

For each class $k \in \{0, ..., K-1\}$:

* $x \mid y = k \sim \mathcal{N}(\mu_k, \Sigma_k)$
* $y \sim \text{Categorical}(\phi_0, ..., \phi_{K-1})$


## Model Derivation

### Posterior using Bayes' Theorem:

$$
P(y = k \mid x) = \frac{P(x \mid y = k) P(y = k)}{\sum_{j=0}^{K-1} P(x \mid y = j) P(y = j)}
$$

We predict:

$$
\boxed{h(x) = \arg\max_k \left[ \log P(x \mid y = k) + \log \phi_k \right]}
$$

## Linear Discriminant Analysis (LDA)

### Assumptions

* All class-conditional distributions share the same covariance matrix: $\Sigma_k = \Sigma$
* Only the class means $\mu_k$ differ.

### Decision Function

Each class has a discriminant function:

$$
g_k(x) = x^\top \Sigma^{-1} \mu_k - \frac{1}{2} \mu_k^\top \Sigma^{-1} \mu_k + \log \phi_k
$$

Predict:

$$
\boxed{h(x) = \arg\max_k \, g_k(x)}
$$

### Parameter Estimation (MLE)

From data, estimate:

* **Class priors**:

  $$
  \phi_k = \frac{n_k}{n}, \quad \text{where } n_k = \sum_{i=1}^n \mathbb{1}\{y_i = k\}
  $$

* **Class means**:

  $$
  \mu_k = \frac{1}{n_k} \sum_{i: y_i = k} x_i
  $$

* **Shared covariance matrix**:

  $$
  \Sigma = \frac{1}{n} \sum_{k=0}^{K-1} \sum_{i: y_i = k} (x_i - \mu_k)(x_i - \mu_k)^\top
  $$


## Quadratic Discriminant Analysis (QDA)

### Assumptions

* Each class has its own covariance matrix: $\Sigma_k$
* This yields quadratic decision boundaries.

### Decision Function

Each class has a quadratic discriminant:

$$
g_k(x) = -\frac{1}{2} \log |\Sigma_k| - \frac{1}{2}(x - \mu_k)^\top \Sigma_k^{-1}(x - \mu_k) + \log \phi_k
$$

Predict:

$$
\boxed{h(x) = \arg\max_k \, g_k(x)}
$$

### Parameter Estimation (MLE)

For each class $k$:

* **Class prior**:

  $$
  \phi_k = \frac{n_k}{n}
  $$

* **Class mean**:

  $$
  \mu_k = \frac{1}{n_k} \sum_{i: y_i = k} x_i
  $$

* **Class-specific covariance**:

  $$
  \Sigma_k = \frac{1}{n_k} \sum_{i: y_i = k} (x_i - \mu_k)(x_i - \mu_k)^\top
  $$

## Comparison: LDA vs QDA

| Feature               | LDA                                       | QD                                           |
| --------------------- | ----------------------------------------- | -------------------------------------------- |
| Covariance Assumption | Same for all classes                      | Unique per class                             |
| Decision Boundary     | Linear                                    | Quadratic                                    |
| Parameters Estimated  | $\mu_k, \phi_k, \Sigma$                   | $\mu_k, \phi_k, \Sigma_k$                    |
| Complexity            | Simpler, lower variance                   | More flexible, but can overfit on small data |
| Ideal Use Case        | Classes share similar spread (covariance) | Classes have distinct covariance structures  |


## Conclusion

Multiclass GDA extends the binary case by modeling each class $k \in \{0, ..., K-1\}$ with its own mean and covariance (for QDA) or a shared covariance (for LDA). The model remains clear and interpretable:

* LDA gives a linear classifier, computationally efficient and robust.
* QDA is more expressive, better when covariances differ.

Both compute closed-form MLE solutions, making them fast and useful even in high dimensions.
