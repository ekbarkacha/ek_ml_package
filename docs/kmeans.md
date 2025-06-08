# K-Means Clustering

## Table of Contents

1. [Introduction](#introduction)
2. [Problem Setup](#problem-setup)
3. [Objective Function](#objective-function)
4. [K-Means Algorithm](#k-means-algorithm)
5. [Mathematical Derivation](#mathematical-derivation)
6. [Convergence and Complexity](#convergence-and-complexity)
7. [Limitations and Considerations](#limitations-and-considerations)
8. [Conclusion](#conclusion)

---

## Introduction

K-Means is an unsupervised learning algorithm that partitions a dataset into K clusters, where each point belongs to the cluster with the nearest mean. It aims to minimize intra-cluster variance while maximizing inter-cluster separation. The algorithm is simple yet powerful and it’s widely used in data mining and pattern recognition.

## Problem Setup

Let:

* $n$: number of data points
* $d$: number of dimensions/features
* $k$: number of clusters

We define:

* $\mathbf{X} \in \mathbb{R}^{n \times d}$: data matrix where each row $x_i \in \mathbb{R}^d$
* $\mu_1, \mu_2, \dots, \mu_k \in \mathbb{R}^d$: centroids of clusters
* $C_i$: the set of data points assigned to cluster $i$ for $i = 1, 2, ..., k$

## Objective Function

K-means aims to minimize the within-cluster sum of squares (WCSS), also called the distortion function:

$$
J = \sum_{i=1}^{k} \sum_{x \in C_i} \|x - \mu_i\|^2
$$

The goal is to find the cluster assignments $C_i$ and the centroids $\mu\_i$ that minimize $J$.

## K-Means Algorithm

The algorithm follows an iterative refinement approach:

**Input:** Dataset $\mathbf{X}$, number of clusters $k$
**Output:** Set of centroids $\mu_1, \dots, \mu_k$, cluster assignments

**Steps:**

1. **Initialization**: Randomly choose $k$ points from the dataset as initial centroids.
2. **Assignment Step**: Assign each data point to the cluster whose centroid is closest:

$$
C_i = \\{ x_j : \|x_j - \mu_i\|^2 \le \|x_j - \mu_l\|^2 \quad \forall l = 1, \dots, k \\}
$$
3. **Update Step**: Recompute the centroids as the mean of the assigned points:

   $$
   \mu_i = \frac{1}{|C_i|} \sum_{x \in C_i} x
   $$
4. **Repeat** steps 2 and 3 until convergence (no change in assignments or centroids).

## Mathematical Derivation

### 1. **Assignment Step Minimizes Distance**

We fix the centroids $\mu_1, \dots, \mu_k$ and assign each $x_j$ to the nearest centroid. For each point, we solve:

$$
\min_{i \in \{1,\dots,k\}} \|x_j - \mu_i\|^2
$$

This step ensures that, for a fixed set of centroids, the loss $J$ is minimized with respect to cluster assignments.

### 2. **Update Step Minimizes WCSS for Fixed Assignments**

Fixing the assignments $C_1, \dots, C_k$, the best centroid $\mu_i$ that minimizes:

$$
\sum_{x \in C_i} \|x - \mu_i\|^2
$$

is obtained by solving:

$$
\frac{d}{d\mu_i} \sum_{x \in C_i} \|x - \mu_i\|^2 = 0
$$

Using vector calculus:

$$
\begin{aligned}
\|x - \mu_i\|^2 &= (x - \mu_i)^\top(x - \mu_i) \\
\frac{d}{d\mu_i} \|x - \mu_i\|^2 &= -2(x - \mu_i) \\
\Rightarrow \frac{d}{d\mu_i} \sum_{x \in C_i} \|x - \mu_i\|^2 &= -2 \sum_{x \in C_i} (x - \mu_i)
\end{aligned}
$$

Setting the derivative to 0:

$$
\sum_{x \in C_i} (x - \mu_i) = 0 \Rightarrow \mu_i = \frac{1}{|C_i|} \sum_{x \in C_i} x
$$


## Convergence and Complexity

* **Convergence**: K-Means is guaranteed to converge to a **local minimum** of the objective $J$, because each step (assignment and update) does not increase the objective.
* **Time Complexity**:

  * One iteration: $\mathcal{O}(nkd)$
  * If the algorithm converges in $T$ iterations, total: $\mathcal{O}(nkdT)$


## Limitations and Considerations

| Limitation                     | Description                                                          |
| ------------------------------ | -------------------------------------------------------------------- |
| **Local Minimum**              | May converge to different solutions depending on initialization      |
| **Number of Clusters ($k$)** | Must be specified beforehand                                         |
| **Spherical Clusters Assumed** | Works best when clusters are isotropic (round, same size)            |
| **Sensitive to Outliers**      | Outliers can distort the position of centroids                       |
| **Empty Clusters**             | A cluster may become empty during iterations, requiring reassignment |

### Initialization Strategies

To address random initialization issues, methods like K-Means++ are used, which spread out initial centroids:

$$
P(x) \propto \min_{i} \|x - \mu_i\|^2
$$

#### K-Means++ Initialization (with Softmax Probabilities)

K-Means++ improves the standard K-Means algorithm by carefully choosing initial centroids to be well-separated, which often leads to improved convergence and clustering quality. This version uses the softmax function to transform distances into selection probabilities.

**Initialize Centroids using K-Means++:**

1. **Choose the first centroid** randomly from the dataset $\mathcal{X} = {x_1, x_2, \dots, x_n}$.

2. **For each data point** $x_i \in \mathcal{X}$ that is not yet selected as a centroid, compute the distance $d(x_i)$ to the nearest already chosen centroid:

      \(
      d(x_i) = \min_{\mu \in \{\mu_1, \dots, \mu_t\}} \|x_i - \mu\|
      \)

3. **Convert distances to probabilities** using the softmax function:

      \(
      p(x_i) = \frac{e^{d(x_i)}}{\sum_{j=1}^{n} e^{d(x_j)}}
      \)

      This assigns a higher probability to points that are farther from existing centroids, but in a smooth, differentiable manner.

4. **Select the next centroid randomly** from the dataset using the probability distribution $p(x_i)$.

5. **Repeat steps 2–4** until $k$ centroids have been initialized.

6. Continue with the standard K-Means algorithm using these initial centroids.


## Conclusion

K-Means is an important clustering algorithm that minimizes intra-cluster variance using an iterative algorithm. The two main steps—assignment and update—each reduce the objective, ensuring convergence. While simple and efficient, it assumes spherical clusters and requires choosing the number of clusters beforehand.