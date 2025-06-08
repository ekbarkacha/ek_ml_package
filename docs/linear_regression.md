# Linear Regression And Gradient Descent

## Table of Contents
1. [Introduction](#introduction)
2. [Data Representation](#data-representation)
3. [Model Hypothesis](#model-hypothesis)
4. [Loss Function (L2 Norm)](#loss-function)
4. [Gradient Of Loss Function](#gradient)
5. [Gradient Descent Algorithms](#gradient-descent-algorithms)
    - Batch Gradient Descent
    - Stochastic Gradient Descent
    - Stochastic Gradient Descent with Momentum
    - Mini-batch Gradient Descent
6. [Conclusion](#conclusion)

## Introduction

Linear regression models a linear relationship between input features **X** and a target vector **y**. Using matrix operations simplifies computation and scales better for large datasets.

---


## Data Representation

Let:

- $n$: number of training examples  
- $d$: number of features  

We define:

- $\mathbf{X} \in \mathbb{R}^{n \times d}$: feature matrix  
- $\mathbf{y} \in \mathbb{R}^{n \times 1}$: target vector  
- $\beta \in \mathbb{R}^{d \times 1}$: weight vector

$$
\mathbf{X} =
\begin{bmatrix}
x_1^{(1)} & x_2^{(1)} & \dots & x_d^{(1)} \\
x_1^{(2)} & x_2^{(2)} & \dots & x_d^{(2)} \\
\vdots & \vdots & \ddots & \vdots \\
x_1^{(n)} & x_2^{(n)} & \dots & x_d^{(n)}
\end{bmatrix}
$$

---

## Model Hypothesis

Given the dataset in the form of $(x_i, y_i)$ for $i = 1, 2, \dots, n$, and under the following assumptions:

- $y_i = \beta x_i + \epsilon_i$ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *(Linearity assumption)*  
- $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$ &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; *(Noise is normally distributed with zero mean)*

Then the expected value of $y_i$ conditioned on $x_i$ is:

$$
\begin{aligned}
\hat{y}_i &= \mathbb{E}[y_i \mid x_i] \\
         &= \mathbb{E}[\beta x_i + \epsilon_i] \\
         &= \beta x_i + \mathbb{E}[\epsilon_i] \\
         &= \beta x_i
\end{aligned}
$$

Hence, the predicted output is a linear function of the input.

### Matrix Form

The model hypothesis in matrix form is:

$$
\hat{\mathbf{y}} = \mathbf{X} \beta
$$

Where:

- $\mathbf{X} \in \mathbb{R}^{n \times d}$ is the feature matrix  
- $\beta \in \mathbb{R}^{d \times 1}$ is the weight vector  
- $\hat{\mathbf{y}} \in \mathbb{R}^{n \times 1}$ is the vector of predictions

---

## Loss Function

To derive the loss function for linear regression, we start by modeling the data generation process probabilistically.

### Assumption: Gaussian Noise Model

We assume that the observed outputs $y_i$ are generated from a linear model with additive Gaussian noise:

$$
\begin{aligned}
y_i &= \mathbf{x}_i \beta + \epsilon_i, \quad \epsilon_i \sim \mathcal{N}(0, \sigma^2)\\
\hat{y}_i&= \mathbf{x}_i \beta
\end{aligned}
$$


Then, the conditional probability of observing $y_i$ given $\mathbf{x}_i$ and the parameters $\beta$ is:

$$
p(y_i \mid \mathbf{x}_i; \beta) = \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left( -\frac{(y_i - \mathbf{x}_i \beta)^2}{2\sigma^2} \right)
$$

### Likelihood 


$$
\mathcal{L}(\beta) = \prod_{i=1}^n p(y_i \mid \mathbf{x}_i; \beta) = \prod_{i=1}^n \frac{1}{\sqrt{2\pi \sigma^2}} \exp\left( -\frac{(y_i - \mathbf{x}_i \beta)^2}{2\sigma^2} \right)
$$

### Log-Likelihood

To simplify, take the **log-likelihood**:

$$
\log \mathcal{L}(\beta) = -\frac{n}{2} \log(2\pi \sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^n (y_i - \mathbf{x}_i\beta)^2
$$

### Maximum Log-likelihood Estimator (MLE)

$$\hat{\beta} = \arg\max_{\beta} \left( -\frac{n}{2} \log(2\pi \sigma^2) - \frac{1}{2\sigma^2} \sum_{i=1}^n (y_i - \mathbf{x}_i \beta)^2 \right)$$


Since $\sigma^2$ is constant and we want to **maximize the log-likelihood**, this is equivalent to **minimizing the negative log-likelihood** (NLL):

$$
\begin{aligned}
\hat{\beta} &= \arg\min_{\beta} \left(\frac{1}{2\sigma^2} \sum_{i=1}^n (y_i - \mathbf{x}_i \beta)^2\right)\\
&= \arg\min_{\beta} \left(\frac{1}{2\sigma^2} \sum_{i=1}^n (y_i - \hat{y}_i)^2\right) \text{ since }\hat{y}_i&= \mathbf{x}_i \beta
\end{aligned}
$$

Ignoring constants, minimizing NLL is equivalent to minimizing the **Mean Squared Error (MSE)**:

$$
\hat{\beta} = \arg\min_{\beta} \left(\sum_{i=1}^n (y_i - \hat{y}_i)^2\right)
$$

### Final Loss Function (MSE)

In matrix form:

$$
\mathcal{L}(\beta) = \frac{1}{n} \| \mathbf{y} - \mathbf{X} \beta\|_2^2
$$

Where:

- $\mathbf{X} \in \mathbb{R}^{n \times d}$
- $\mathbf{y} \in \mathbb{R}^{n \times 1}$
- $\beta \in \mathbb{R}^{d \times 1}$

---

## Gradient of the Loss Function

We begin with the **Mean Squared Error (MSE)** loss function in matrix form as shown above.

$$
\begin{aligned}
\mathcal{L}(\beta) &= \frac{1}{n} \| \mathbf{y} - \mathbf{X} \beta\|_2^2\\
                   &= \frac{1}{n} (\mathbf{y} - \mathbf{X} \beta)^\top (\mathbf{y} - \mathbf{X} \beta)
\end{aligned}
$$

Let’s expand the inner product:

$$
\begin{aligned}
\mathcal{L}(\beta)
&= \frac{1}{n} \left( \mathbf{y}^\top \mathbf{y} - 2 \mathbf{y}^\top \mathbf{X} \beta + \beta^\top \mathbf{X}^\top \mathbf{X} \beta \right)
\end{aligned}
$$

Now, we compute the gradient of each term with respect to $\beta$:

- $\nabla_\beta(\mathbf{y}^\top \mathbf{y}) = 0$ (no $\beta$ involved)  
- $\nabla_\beta(-2 \mathbf{y}^\top \mathbf{X} \beta) = -2 \mathbf{X}^\top \mathbf{y}$  
- $\nabla_\beta(\beta^\top \mathbf{X}^\top \mathbf{X} \beta) = 2 \mathbf{X}^\top \mathbf{X} \beta$

Putting it all together:

$$
\nabla_\beta \mathcal{L}(\beta) = \frac{1}{n} \left( -2 \mathbf{X}^\top \mathbf{y} + 2 \mathbf{X}^\top \mathbf{X} \beta \right)
$$

To simplify the expression we factor out the 2 which gives the gradient of the MSE loss with respect to $\beta$:

$$
\nabla_\beta \mathcal{L}(\beta) = \frac{2}{n} \left( \mathbf{X}^\top \mathbf{X} \beta - \mathbf{X}^\top \mathbf{y} \right)
$$

This gradient is used to perform parameter updates in **Gradient Descent**:

$$
\beta_1 = \beta_0 - \alpha \cdot \nabla_\beta \mathcal{L}(\beta_0)
$$

Where:

- $\alpha$ is the learning rate

## Solving for $\beta$ (Normal Equation)

We can find the optimal $\beta$ by setting the gradient of the loss function to zero:

$$
\nabla_\beta \mathcal{L}(\beta) = \frac{2}{n} \left( \mathbf{X}^\top \mathbf{X} \beta - \mathbf{X}^\top \mathbf{y} \right) = 0
$$

Multiplying both sides by $\frac{n}{2}$:

$$
\mathbf{X}^\top \mathbf{X} \beta = \mathbf{X}^\top \mathbf{y}
$$

Solving for $\beta$ gives the **closed-form solution** (also called the **normal equation**):

$$
\hat{\beta} = (\mathbf{X}^\top \mathbf{X})^{-1} \mathbf{X}^\top \mathbf{y}
$$

### Issue: Non-Invertibility of $\mathbf{X}^\top \mathbf{X}$

In practice, the matrix $\mathbf{X}^\top \mathbf{X}$ may not be invertible if:

- Features are **linearly dependent** (i.e., multicollinearity)
- The number of features $d$ is **greater than** the number of samples $n$ (underdetermined system)
- Numerical precision issues due to **ill-conditioning**

In these cases, we cannot compute \( \hat{\beta} \) directly using the normal equation.


### Solution: Add Regularization (Ridge Regression)

To address non-invertibility and reduce overfitting, we add an **L2 regularization term** (also known as **Ridge Regression**):

Modified loss function:

$$
\mathcal{L}_{\text{ridge}}(\beta) = \frac{1}{n} \| \mathbf{y} - \mathbf{X} \beta \|_2^2 + \lambda \| \beta \|_2^2
$$

Where:

- $\lambda > 0$ is the regularization strength
- $\| \beta \|_2^2 = \beta^\top \beta$

### Closed-Form Solution with Regularization

Taking the gradient and setting it to zero:

$$
\nabla_\beta \mathcal{L}_{\text{ridge}}(\beta) = \frac{2}{n} \mathbf{X}^\top (\mathbf{X} \beta - \mathbf{y}) + 2 \lambda \beta = 0
$$

Leads to the regularized normal equation:

$$
\hat{\beta}_{\text{ridge}} = \frac{1}{n} \left( (\mathbf{X}^\top \mathbf{X} + \lambda \mathbf{I})^{-1} \mathbf{X}^\top \mathbf{y} \right)
$$

This ensures invertibility even when $\mathbf{X}^\top \mathbf{X}$ is singular, due to the identity matrix $\mathbf{I}$ added to it.

---

## Gradient Descent Algorithms 

Gradient Descent is an iterative optimization algorithm used to minimize the loss function by updating the parameters in the direction of the negative gradient.

We update the parameter vector $\beta$ using the rule:

$$
\beta = \beta - \alpha \cdot \nabla_\beta \mathcal{L}(\beta)
$$

Where:
- $\alpha$ is the learning rate
- $\nabla_\beta \mathcal{L}(\beta)$ is the gradient of the loss function with respect to $\beta$

In this section, we describe three variations of the **Gradient Descent** (GD) algorithm used for training a linear regression model: **Batch Gradient Descent (BGD)**, **Stochastic Gradient Descent (SGD)**, and **Mini-Batch Gradient Descent (MBGD)**.


---

## 1. Batch Gradient Descent

Batch Gradient Descent computes the gradient using the **entire dataset** and updates the parameters at the end of each epoch.

### **Input:**  
- Dataset: $D = \{ (\mathbf{x}_i, y_i) \}_{i=1}^{n}$ 
- Learning rate: $\epsilon$  
- Number of epochs  
- Tolerance: `tol`

### **Output:**  
- Optimized weight vector $\beta$

### **Algorithm:**

```text
1. Initialize β₀ = 0 ∈ ℝᵈ
2. for epoch in number of epochs do:
    3. Shuffle the dataset D
    4. Compute predictions: ŷ = X β₀
    5. Compute loss: L = (1/n) ‖ŷ − y‖²
    6. Compute gradient: ∇βf(β₀) = (2/n)  (XᵀXβ₀ − Xᵀy)
    7. Update weights: β₁ = β₀ − ε ∇βf(β₀)
    8. if ‖y − X β₁‖² < tol then:
        9. break
    10. β₀ = β₁

```


## 2. Stochastic Gradient Descent

Stochastic Gradient Descent updates the weights after processing each individual **data point**. This approach tends to converge faster, but the updates are noisy.

### **Input:**  
- Dataset: $D = \{ (\mathbf{x}_i, y_i) \}_{i=1}^{n}$  
- Learning rate: $\epsilon$  
- Number of epochs  
- Tolerance: `tol`

### **Output:**  
- Optimized weight vector $\beta$

### **Algorithm:**

```text
1. Initialize β₀ = 0 ∈ ℝᵈ
2. for epoch in number of epochs do:
    3. Shuffle the dataset D
    4. for i = 1 to n do:
        5. Compute prediction for observation i: ŷᵢ = xᵢᵀ β₀
        6. Compute loss: L = (ŷᵢ − yᵢ)²
        7. Compute gradient: ∇βf(β₀) = 2 ( xᵢxᵢᵀ β₀ − xᵢyᵢ ) 
        8. Update weights: β₁ = β₀ − ε ∇βf(β₀)
        9. if (ŷᵢ − yᵢ)² < tol then:
            10. break
    11. β₀ = β₁
```

---

## 3. Stochastic Gradient Descent with Momentum

**Momentum** is an enhancement to standard SGD that helps accelerate gradient descent in the relevant direction and dampens oscillations. It is particularly useful in scenarios where the optimization landscape has high curvature, small but consistent gradients, or noisy updates.

### **Motivation:**

- **Variance in SGD**: Helps smooth out noisy gradients by using an exponentially decaying moving average.
- **Poor conditioning**: Helps accelerate through narrow valleys and avoids getting stuck.

### **Concept:**

We introduce a **velocity vector** $v$ that accumulates the exponentially decaying average of past gradients:

- $v$ is initialized to zero
- $\beta$ is the **momentum coefficient** (typically $\beta \in [0.9, 0.99]$)
- $\epsilon$ is the learning rate

### **Update Rule:**

$$
v_t = \beta v_{t-1} + (1 - \beta) \nabla_\beta \mathcal{L}(\beta_{t-1})
$$

$$
\beta_t = \beta_{t-1} - \epsilon v_t
$$

This adds "inertia" to the parameter updates, similar to how momentum works in physics.

---

### **Algorithm:**

**Input:**

- Dataset: $D = \{ (\mathbf{x}_i, y_i) \}_{i=1}^{n}$  
- Learning rate: $\epsilon$  
- Momentum coefficient: $\beta$  
- Number of epochs  
- Tolerance: `tol`

**Output:**

- Optimized weight vector $\beta$

```text
1. Initialize β₀ = 0 ∈ ℝᵈ
2. Initialize velocity vector v = 0 ∈ ℝᵈ
3. for epoch in number of epochs do:
    4. Shuffle the dataset D
    5. for i = 1 to n do:
        6. Compute prediction for observation i: ŷᵢ = xᵢᵀ β₀
        7. Compute gradient: ∇βf(β₀) = 2 ( xᵢxᵢᵀ β₀ − xᵢyᵢ ) 
        8. Update velocity: v = β ⋅ v + (1 - β) ⋅ ∇βf(β₀)
        9. Update weights: β₁ = β₀ − ε ⋅ v
        10. if (ŷᵢ − yᵢ)² < tol then:
            11. break
    12. β₀ = β₁
```

---

## 4. Mini-Batch Gradient Descent

Mini-Batch Gradient Descent updates the weights after processing small **batches** of data, offering a balance between the computational efficiency of BGD and the faster convergence of SGD.

### **Input:**  
- Dataset: $D = \{ (\mathbf{x}_i, y_i) \}_{i=1}^{n}$  
- Learning rate: $\epsilon$ 
- Number of epochs  
- Batch size: $B$
- Tolerance: `tol`

### **Output:**  
- Optimized weight vector $\beta$

### **Algorithm:**

```text
1. Initialize β₀ = 0 ∈ ℝᵈ
2. for epoch in number of epochs do:
    3. Shuffle the dataset D
    4. Partition data into n_batches = ceil(n / B)
    5. for j = 1 to n_batches do:
        6. Get mini-batch (Xⱼ, yⱼ)
        7. Compute prediction: ŷⱼ = Xⱼ β₀
        8. Compute loss: L = (1/B) ‖ŷⱼ − yⱼ‖²
        9. Compute gradient: ∇βf(β₀) = (2/B)  (XⱼᵀXⱼ β₀ − Xⱼᵀyⱼ)
        10. Update weights: β₁ = β₀ − ε ∇βf(β₀)
        11. if ‖yⱼ − Xⱼ β₁‖² < tol then:
            12. break
    13. β₀ = β₁
```

---


## Conclusion
Linear Regression is foundational in machine learning and serves as the basis for more complex models. By understanding how to train it using different optimization methods like GD, SGD, and mini-batch GD, one gains deep insights into how models learn from data.