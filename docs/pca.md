# Principal Component Analysis (PCA)

## Table of Contents

1. [Introduction](#introduction)
2. [Standardization](#standardization)
3. [Covariance Matrix](#covariance-matrix)
4. [Eigen Decomposition](#eigen-decomposition)
5. [Dimensionality Reduction](#dimensionality-reduction)
6. [Algorithm](#algorithm)
7. [Reconstruction](#reconstruction)
8. [Conclusion](#conclusion)

## Introduction

Principal Component Analysis (PCA) is a linear dimensionality reduction technique that finds the directions (principal components) that capture the most variance in the data. PCA transforms the data into a new coordinate system where the axes (principal components) are ordered by the amount of variance they explain.

## Standardization

Before applying PCA, we standardize the data so that each feature contributes equally, especially when features are on different scales.

Let:

* $\mathbf{X} \in \mathbb{R}^{n \times d}$ be the original data matrix
* $x_{ij}$ is the value of the $j$-th feature for the $i$-th sample

### Compute Mean and Standard Deviation

Let:

* $\mu_j = \frac{1}{n} \sum_{i=1}^{n} x_{ij}$
* $\sigma_j = \sqrt{ \frac{1}{n} \sum_{i=1}^{n} (x_{ij} - \mu_j)^2 }$

### Standardize Each Feature

$$
x_{ij}^{\text{(std)}} = \frac{x_{ij} - \mu_j}{\sigma_j}
$$

Let:

* $\mathbf{X}\_{\text{std}}$ be the standardized data matrix

So now:

$$
\mathbf{X}_{\text{std}} = \frac{\mathbf{X} - \mu}{\sigma}
$$

where $\mu$ and $\sigma$ are broadcast row-wise over all samples

Now $\mathbf{X}_{\text{std}}$ has zero mean and unit variance per feature.

## Covariance Matrix

The covariance matrix measures how features vary with respect to each other:

$$
\mathbf{\Sigma} = \frac{1}{n} \mathbf{X}_{\text{std}}^\top \mathbf{X}_{\text{std}} \in \mathbb{R}^{d \times d}
$$

* Diagonal elements represent feature variances
* Off-diagonal elements represent covariances

## Eigen Decomposition

To find the principal components, we compute the eigenvectors and eigenvalues of $\mathbf{\Sigma}$:

$$
\mathbf{\Sigma} v_i = \lambda_i v_i
$$

Where:

* $v_i$ = $i$-th eigenvector (principal direction)
* $\lambda_i$ = variance explained along $v_i$

Sort eigenvalues: $\lambda_1 \ge \lambda_2 \ge \cdots \ge \lambda_d$

Let:

* $\mathbf{V} = [v_1, v_2, \dots, v_d] \in \mathbb{R}^{d \times d}$
* $\mathbf{\Lambda} = \text{diag}(\lambda_1, \dots, \lambda_d)$

Then:

$$
\mathbf{\Sigma} = \mathbf{V} \mathbf{\Lambda} \mathbf{V}^\top
$$

## Dimensionality Reduction

To reduce the dataset from $d$ dimensions to $k$ dimensions ($k < d$), we take the top $k$ eigenvectors:

Let:

* $\mathbf{V} = [v_1, \dots, v_k] \in \mathbb{R}^{d \times k}$

Then the projected data is:

$$
\mathbf{Z} = \mathbf{X}_{\text{std}} \mathbf{V} \in \mathbb{R}^{n \times k}
$$

Each row in $\mathbf{Z}$ is the lower-dimensional representation of the original sample.


## Algorithm

**Input:**

* Dataset: $D = \{ \mathbf{x}_i \}_{i=1}^n$, where $\mathbf{x}_i \in \mathbb{R}^d$
* Desired number of components: $k$ such that $k < d$
* Standardization flag: `standardize = True/False`

**Output:**

* Projected data $\mathbf{Z} \in \mathbb{R}^{n \times k}$
* Projection matrix $\mathbf{V} \in \mathbb{R}^{d \times k}$

```text
1. Compute the mean vector μ = (1/n) ∑ xᵢ ∈ ℝᵈ
2. Center the data: X_centered = X − 1⋅μᵀ  ∈ ℝⁿˣᵈ

3. If standardize is True then:
    4. Compute std deviation vector σ = sqrt((1/n) ∑ (xᵢ − μ)²)
    5. Standardize: X_std = X_centered / σ  (element-wise)
    6. X ← X_std
   else:
    7. X ← X_centered

8. Compute covariance matrix: Σ = (1/n) ⋅ XᵀX ∈ ℝᵈˣᵈ

9. Compute eigenvalues λ₁,...,λ_d and eigenvectors v₁,...,v_d of Σ
10. Sort eigenvectors V = [v₁, ..., v_d] by descending eigenvalues

11. Form projection matrix: V = [v₁, ..., v_k] ∈ ℝᵈˣᵏ

12. Project data onto top-k components: Z = X ⋅ V ∈ ℝⁿˣᵏ

13. Return Z, V
```

## Reconstruction

To reconstruct the approximation of the original standardized data from the reduced representation:

$$
\hat{\mathbf{X}}_{\text{std}} = \mathbf{Z} \mathbf{V}^\top
$$

To return to the original scale, we unstandardize:

$$
\hat{\mathbf{X}} = \hat{\mathbf{X}}_{\text{std}} \cdot \sigma + \mu
$$

where $\sigma$ and $\mu$ are broadcast appropriately across dimensions

## Conclusion

PCA is a powerful unsupervised technique to:

* Reduce dimensionality
* Remove noise
* Visualize high-dimensional data


### Summary Table

| Step                    | Description                                                   |
| ----------------------- | ------------------------------------------------------------- |
| **Standardization**     | Ensure each feature has zero mean and unit variance           |
| **Covariance Matrix**   | Compute feature-wise covariance after standardization         |
| **Eigen Decomposition** | Extract eigenvectors and eigenvalues of the covariance matrix |
| **Projection**          | Project original data onto top $k$ eigenvectors               |
| **Reconstruction**      | Approximate the original data using the reduced components    |
