# K-Nearest Neighbors (KNN)

## Table of Contents

1. [Introduction](#introduction)
2. [Data Representation](#data-representation)
3. [Model Hypothesis](#model-hypothesis)
4. [Loss Function](#loss-function)
5. [Classification Algorithm](#classification-algorithm)
6. [Distance Metrics](#distance-metrics)
7. [Advantages and Disadvantages](#advantages-and-disadvantages)
8. [Conclusion](#conclusion)

---

## Introduction

K-Nearest Neighbors (KNN) is a non-parametric and lazy learning algorithm used for classification and regression tasks. It does not learn an explicit model during training and instead it stores the training dataset and uses it during the prediction time.

* **Non-Parametric**: KNN does not assume any underlying distribution for the data.
* **Lazy Learning**: The algorithm does not train on the dataset immediately but instead performs all computation during prediction.

### Basic Idea:

For a given test point, KNN looks at the **K closest training points** (neighbors) in the feature space and outputs the most common class (for classification) or average (for regression) of these neighbors.

---

## Data Representation

Let:

* $n$: number of training examples
* $d$: number of features

We define:

* $\mathbf{X} \in \mathbb{R}^{n \times d}$: feature matrix (training data)
* $\mathbf{y} \in {0, 1}^n$: target vector (labels) for classification tasks (can be continuous for regression)
* $\mathbf{x\_{\text{test}}} \in \mathbb{R}^{1 \times d}$: feature vector for the test point

In classification tasks, we assume that each data point is labeled as either $y_i \in {0, 1}$ (binary classification), or can be from a set of multiple classes for multiclass classification.

---

## Model Hypothesis

In KNN, the hypothesis is simply based on local neighborhoods. Given a new test point $\mathbf{x\_{\text{test}}}$, the model classifies it by looking at the K nearest neighbors from the training set $\mathbf{X}$. The output is the most common class among these neighbors.

* **Classification**: The label for the test point is predicted based on the majority vote of the K nearest neighbors:

$$
\hat{y}_{\text{test}} = \text{mode}(y_{\text{neighbors}}) \quad \text{where } y_{\text{neighbors}} \text{ is the label of the K nearest neighbors}.
$$

* **Regression**: The predicted value is the average of the K nearest neighbors' values:

$$
\hat{y}_{\text{test}} = \frac{1}{K} \sum_{i=1}^{K} y_i
$$

### Key Concept:

KNN relies on the assumption that similar data points (i.e., those close in the feature space) share similar labels.

---

## Loss Function

KNN is not trained with a traditional loss function like in supervised models such as Logistic Regression. Instead, the "loss" comes from the accuracy of the model during prediction. However, for understanding and performance evaluation, the accuracy or error rate of the predictions can be considered the equivalent of a loss function.

### For Classification:

* **Accuracy** (as a performance measure):

  $$
  \text{Accuracy} = \frac{\text{Number of Correct Predictions}}{n_{\text{test}}}
  $$

  * Where $n\_{\text{test}}$ is the number of test examples.
* **Misclassification Rate** (as the loss function):

  $$
  \text{Loss} = \frac{\text{Number of Incorrect Predictions}}{n_{\text{test}}}
  $$

### For Regression:

* **Mean Squared Error (MSE)** (Loss Function):

  $$
  \text{MSE} = \frac{1}{n_{\text{test}}} \sum_{i=1}^{n_{\text{test}}} (y_{\text{true}}^{(i)} - \hat{y}_{\text{test}}^{(i)})^2
  $$

  * Where $y_{\text{true}}$ is the true value, and $\hat{y}_{\text{test}}$ is the predicted value for test point $i$.

---

## Classification Algorithm

1. **Training Phase**:

    - The KNN algorithm does not perform explicit training. All data points are stored in memory.

2. **Prediction Phase**:

    - Given a test point $\mathbf{x_{\text{test}}}$, find the K nearest neighbors from the training set.
    - The distance between $\mathbf{x_{\text{test}}}$ and every point in the training set is calculated.
    - Sort the training points by distance.
    - Select the top K nearest neighbors.
    - For classification, perform a majority vote among the K neighbors to determine the predicted class.

---

## Distance Metrics

The distance between points is very important in KNN, as it directly affects the algorithm's performance. The most commonly used distance metrics are:

1. **Euclidean Distance** (or L2 Distance):

    \(
    d(\mathbf{x}_i, \mathbf{x}_j) = \sqrt{\sum_{k=1}^{d} (x_{i,k} - x_{j,k})^2}
    \)

2. **Manhattan Distance** (or L1 Distance):

      \(
      d(\mathbf{x}_i, \mathbf{x}_j) = \sum_{k=1}^{d} |x_{i,k} - x_{j,k}|
      \)

3. **Cosine Similarity** (often used for text data):

      \(
      d(\mathbf{x}_i, \mathbf{x}_j) = 1 - \frac{\mathbf{x}_i \cdot \mathbf{x}_j}{\|\mathbf{x}_i\| \|\mathbf{x}_j\|}
      \)

### Choosing K:

* **Small K (e.g., K = 1)**: More sensitive to noise but can model highly localized patterns.
* **Large K**: Less sensitive to noise, but may smooth out important patterns (underfitting).

---

## Advantages and Disadvantages

### Advantages:

* **Simplicity**: KNN is easy to understand and implement.
* **No Training Phase**: KNN is a lazy learner, so it doesn't require training data to build an explicit model.
* **Versatile**: KNN can be used for both classification and regression tasks.

### Disadvantages:

* **Computationally Expensive**: KNN can be slow, especially for large datasets, because it needs to compute the distance between the test point and all points in the training set.
* **Sensitive to Irrelevant Features**: The performance of KNN can degrade if the data has many irrelevant features, as distances become less meaningful.
* **Storage**: KNN requires storing all the training data, which may be infeasible for very large datasets.

---

## Conclusion

The K-Nearest Neighbors (KNN) algorithm is a simple, instance-based learning method that makes predictions based on the proximity of data points. It is widely used due to its simplicity and effectiveness for both classification and regression problems. Despite its simplicity, KNN can be computationally expensive and is sensitive to irrelevant or noisy features. Proper distance metrics and selecting an appropriate K value are essential for achieving good performance with this algorithm.
 