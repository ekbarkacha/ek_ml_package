# Introduction to Machine Learning

This documentation covers the fundamental concepts and theory behind key machine learning algorithms. It is designed to provide clear explanations and mathematical intuition to help deepen your understanding of how these algorithms work.

Use this as a starting point to explore the core ideas and techniques that power many machine learning applications today.


## Table of Contents
- [Introduction](#introduction)
- [Linear Regression](#linear-regression)
- [Linear Model Evaluation](#linear-model-evaluation)
- [K-Fold Cross Validation](#k-fold-cross-validation)
- [Bias-Variance Decomposition](#bias-variance-decomposition)
- [Maximum Likelihood Estimation](#maximum-likelihood-estimation)
- [Examples: MLE Leading to Common Loss Functions](#examples-mle-leading-to-common-loss-functions)
- [Maximum A Posteriori (MAP) Estimation](#maximum-a-posteriori-map-estimation)
- [Gradient Descent from First-Order Taylor Approximation](#gradient-descent-from-first-order-taylor-approximation)
- [Gradient Descent (GD) Convergence](#gradient-descent-gd-convergence)
- [Stochastic Gradient Descent (SGD)](#stochastic-gradient-descent-sgd)
- [Mini-Batch Stochastic Gradient Descent](#mini-batch-stochastic-gradient-descent)
- [Stochastic Gradient Descent (SGD) with Momentum](#stochastic-gradient-descent-sgd-with-momentum)
- [Neural Network](#neural-network)


## Introduction

In supervised machine learning we usually start with the dataset given as:

$$
\mathcal{D} = \left\{ (x_i, y_i) \right\}_{i=1}^{n}
$$

Where $x_i \in \mathcal{X}, y_i \in \mathcal{Y}, (x_i, y_i) \in \mathcal{X} \times \mathcal{Y}$ and $n$ is the number of samples.

$\mathcal{X}$ is known as the input space while $\mathcal{Y}$ is the output space.

If $\mathcal{Y}$ is a finite set of discrete labels then the task is **Classification**. But if $\mathcal{Y} \in \mathbb{R}$ then the task is **Regression**.

In ML we are looking for the best $F$ such that:

$$
F: \mathcal{X} \mapsto \mathcal{Y}
$$

$F$ is known as a **function**, **mapping**, or **hypothesis**. And where $F$ comes from is known as the **hypothesis class/space** $\mathcal{H}$, i.e.,

$$
F \in \mathcal{H}
$$

We also have an objective function (or cost function/criterion) where our main task is to optimize it.

Now we look at one of the common hypothesis spaces known as **linear mapping**, which leads us to **linear regression**.

## Linear Regression

Here our hypothesis is given as:

$$
f(x) = w^\top x
$$

where $x$ is known as the input data and $w$ is the weights (coefficients).

From the above linear mapping, we can have our objective function as:

$$
L(w) = \min_w \;  \left\| w^\top x - y \right\|^2
$$

Now we can find the analytic solution $\hat{w}$ which minimizes our objective function $L(w)$. To solve, we will find the derivative $L(w)$ (gradient) then equate it to zero. Let's solve:

$$
\begin{align*}
    L(w) &= \left\| Xw - Y \right\|^2 \\
         &= (Xw - Y)^\top (Xw - Y) \\
         &= w^\top X^\top X w - w^\top X^\top Y - Y^\top X w + Y^\top Y \\
         &= w^\top X^\top X w - w^\top X^\top Y - Y^\top X w + Y^\top Y
\end{align*}
$$

Now we find the gradient w\.r.t $w$:

$$
\begin{align*}
\frac{d L(w)}{d w} &= X^\top X w + (w^\top X^\top X)^\top - X^\top Y - (Y^\top X)^\top \\
                   &= X^\top X w + X^\top X w - X^\top Y - X^\top Y \\
                   &= 2 X^\top X w - 2 X^\top Y
\end{align*}
$$

Therefore, our gradient $\nabla_w L(w)$ is given as:

$$
\nabla_w L(w) = 2 X^\top X w - 2 X^\top Y
$$

Taking $\nabla_w L(w) = 0$:

$$
\begin{align*}
    & 2 (X^\top X w - X^\top Y) = 0\\
    & X^\top X w =  X^\top Y \\
    & \hat{w} = (X^\top X)^{-1} X^\top Y
\end{align*}
$$

Therefore, the analytical solution $\hat{w}$ that minimizes $L(w)$ is:

$$
\hat{w} = (X^\top X)^{-1} X^\top Y
$$

But the above solution exists only if $X^\top X$ is invertible, i.e., the inverse exists.

If the inverse doesn't exist, we introduce a regularizer. Let's use the $L_2$ regularizer (also known as ridge), where we will have a new objective function given as:

$$
L_{\text{ridge}}(w) = \min_w \;  \left\| Xw - Y \right\|_2^2 + \lambda  \left\| w \right\|_2^2
$$

Where $\lambda > 0$ (positive), since if it's negative and $\left\| w \right\|_2^2$ is too big, then our $L_{\text{ridge}}(w)$ will not be bounded.

Let's find the analytic solution of $L_{\text{ridge}}(w)$:

$$
\begin{align*}
    L_{\text{ridge}}(w) &= \left\| Xw - Y \right\|_2^2 + \lambda  \left\| w \right\|_2^2\\
                        &= (Xw - Y)^\top (Xw - Y) + \lambda w^\top w \\
                        &= w^\top X^\top X w - w^\top X^\top Y - Y^\top X w + Y^\top Y + \lambda w^\top w
\end{align*}
$$

Now we find the gradient of $L_{\text{ridge}}(w)$ w\.r.t $w$:

$$
\begin{align*}
\frac{d L_{\text{ridge}}(w)}{d w} &= X^\top X w + (w^\top X^\top X)^\top - X^\top Y - (Y^\top X)^\top + \lambda w + \lambda w \\
                                  &= X^\top X w + X^\top X w - X^\top Y - X^\top Y + 2\lambda w \\
                                  &= 2 X^\top X w - 2 X^\top Y + 2\lambda w
\end{align*}
$$

Therefore, our gradient $\nabla_w L_{\text{ridge}}(w)$ is given as:

$$
\nabla_w L_{\text{ridge}}(w) = 2 X^\top X w - 2 X^\top Y + 2\lambda w
$$

Taking $\nabla_w L_{\text{ridge}}(w) = 0$:

$$
\begin{align*}
    & 2 (X^\top X w - X^\top Y + \lambda w) = 0\\
    & X^\top X w + \lambda w = X^\top Y\\
    & (X^\top X + \lambda I)w =  X^\top Y \\
    & \hat{w} = (X^\top X + \lambda I)^{-1} X^\top Y
\end{align*}
$$

Therefore, the analytical solution $\hat{w}$ that minimizes $L_{\text{ridge}}(w)$ is:

$$
\hat{w} = (X^\top X + \lambda I)^{-1} X^\top Y
$$

The above solution from ridge regression always exists since $X^\top X + \lambda I$ is always invertible.

## Linear Model Evaluation

For linear regression most time we evaluate our model using mean square error (mse).

Given the dataset:

$$
\mathcal{D} = \left\{ (x_i, y_i) \right\}_{i=1}^{n}
$$

and a model $f(x) = w^\top x$, the **Mean Squared Error (MSE)** is defined as:

$$
Loss_{MSE} = \frac{1}{n} \sum_{i=1}^{n} \left( f(x_i) - y_i \right)^2 = \frac{1}{n} \sum_{i=1}^{n} \left( w^\top x_i - y_i \right)^2
$$

Given the loss/criterion above our aim is to find a function $\hat{f}$ which minimizes our loss i.e
$$
\hat{f} = \arg\min_{f \in \mathcal{H}} \; \frac{1}{n} \sum_{i=1}^{n} \left( f(x_i) - y_i \right)^2
$$ 
where;

* $f$ - hypothesis
* $\mathcal{H}$ - hypothesis class
* $\hat{f}$ - best in $\mathcal{H}$
* $\frac{1}{n} \sum_{i=1}^{n} \left( f(x_i) - y_i \right)^2$ - emperical risk

We have two type of risk:

### 1). **Emperical Risk**
 We get it from our train dataset, which is given as our loss i.e
 $$
 R_{train} = \frac{1}{n} \sum_{i=1}^{n} \left( f(x_i) - y_i \right)^2
 $$

* This is also called **training error** or **empirical risk** $\hat{R}(f)$.
* It’s computable since we have the training dataset $\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n$.

### 2). **True Risk**
$$
 R = \mathbb{E}_{(x,y)\sim \tau} \left[ (f(x) - y)^2 \right]
$$

* $\tau$ is the **true (unknown) data distribution**.
* This is called **expected risk** or **generalization error**.
* We usually can't compute this directly because we don’t know the true distribution $\tau$, only samples from it (the dataset).

### **Why don't we use Empirical Risk to evaluate our model performance?**

This is because **Empirical (Training) Risk** is typically a **biased (optimistic) estimator** of the **True Risk** as it underestimates how well the model will perform on unseen data.

**Proof**

Let $f_{\mathcal{H}}$ be the best-in-class predictor that minimizes the **true risk**, i.e.,

$$
f_{\mathcal{H}} = \arg\min_{f \in \mathcal{H}} R(f) = \arg\min_{f \in \mathcal{H}} \; \mathbb{E}_{(x, y)\sim \tau} \left[ (f(x) - y)^2 \right]
$$

Let $\hat{f}$ be the best-in-class predictor that minimizes the **empirical risk**, i.e.,

$$
\hat{f} = \arg\min_{f \in \mathcal{H}} R_{\text{train}}(f) = \arg\min_{f \in \mathcal{H}} \; \frac{1}{n} \sum_{i=1}^{n} \left( f(x_i) - y_i \right)^2
$$

By definition of $\hat{f}$, we have:

$$
R_{\text{train}}(\hat{f}) \leq R_{\text{train}}(f), \quad \forall f \in \mathcal{H}
$$

Taking the expectation over training samples (drawn from distribution $\tau$):

$$
\mathbb{E}_{\mathcal{D} \sim \tau} \left[ R_{\text{train}}(\hat{f}) \right] \leq \mathbb{E}_{\mathcal{D} \sim \tau} \left[ R_{\text{train}}(f) \right], \quad \forall f \in \mathcal{H}
$$

However, we can go further to have

$$
\mathbb{E} \left[ R_{\text{train}}(\hat{f}) \right] < \mathbb{E} \left[ R(\hat{f}) \right]
$$

That is, the empirical risk of $\hat{f}$ evaluated on the same data it was trained on is generally **less** than the true/generalization risk of $\hat{f}$ on unseen data. Therefore:

$$
R_{\text{train}}(\hat{f}) \text{ is a biased estimate of } R(\hat{f})
$$

### **Since we have seen that Empirical Risk is a biased estimate of the True Risk**, we introduce the concept of the **test dataset** to evaluate our model’s generalization performance.

Given a test dataset:

$$
\mathcal{D}_{\text{test}} = \left\{ (x_i, y_i) \right\}_{i=1}^{k}
$$

Then the **empirical test risk** is defined as:

$$
R_{\text{test}} = \frac{1}{k} \sum_{i=1}^{k} \left( f(x_i) - y_i \right)^2
$$

**Proof:**

$$
\begin{align*}
\mathbb{E} \left[ R_{\text{test}} \right] &= \mathbb{E} \left[ \frac{1}{k} \sum_{i=1}^{k} \left( f(x_i) - y_i \right)^2 \right] \\
                                         &= \frac{1}{k} \sum_{i=1}^{k} \mathbb{E} \left[ \left( f(x_i) - y_i \right)^2 \right] \\
                                         &= \frac{1}{k} \sum_{i=1}^{k} R \\
                                         &= \frac{1}{k} \cdot k \cdot R \\
                                         &= R
\end{align*}
$$

Therefore,
$$
\mathbb{E} \left[ R_{\text{test}} \right] = R
$$

This shows that the **empirical test risk** is an **unbiased estimate** of the **true risk**, assuming the test data is sampled independently from the same distribution $\tau$ as the training data and is not used during training.

Empirical Risk **should not be used** as a performance metric because it’s **biased**. It doesn't reflect how well the model will generalize to new, unseen data — for that, we must estimate the **true risk**, typically using a **validation or test set**.

## K-Fold Cross Validation

K-Fold Cross Validation is a **model evaluation and selection technique** used to estimate how well a model generalizes to unseen data. It is also used to **tune hyperparameters** or **select the best hypothesis** $f \in \mathcal{H}$.

Given a dataset:

$$
\mathcal{D} = \left\{ (x_i, y_i) \right\}_{i=1}^{n}
$$

And a learning algorithm that finds:

$$
\hat{f} = \arg\min_{f \in \mathcal{H}} \; \frac{1}{n} \sum_{i=1}^{n} \left( f(x_i) - y_i \right)^2
$$

We partition the dataset into $k$ folds:

$$
\mathcal{D} = \{ \mathcal{D}_i \}_{i=1}^k
$$

Where:

* $\mathcal{D}_i = \{ (x_j, y_j) \}_{j=1}^{n/k}$
* $\bigcup_{i = 1}^{k} \mathcal{D}_i = \mathcal{D}$
* $\mathcal{D}_i \cap \mathcal{D}_j = \emptyset \quad \forall i \neq j$

### **How K-Fold Cross Validation Works**

1. Split the dataset $\mathcal{D}$ into $k$ approximately equal-sized folds:
   $$
   \mathcal{D}_1, \mathcal{D}_2, \dots, \mathcal{D}_k
   $$

2. For each fold $i \in \{1, 2, \dots, k\}$:
      * **Training set**: $\mathcal{D}_{\text{train}} = \mathcal{D} \setminus \mathcal{D}_i$
      * **Validation set**: $\mathcal{D}_i$
      * Train the model $f_i$ on $\mathcal{D}_{\text{train}}$
      * Evaluate the validation loss on $\mathcal{D}_i$:
      
         \(
         L^{(i)} = \frac{1}{|\mathcal{D}_i|} \sum_{(x, y) \in \mathcal{D}_i} \mathcal{l}(f_i(x), y)
         \)

3. Average the validation losses across all folds:

      \(
      \hat{L}_{\text{cv}} = \frac{1}{k} \sum_{i=1}^{k} L^{(i)}
      \)

### **How Cross Validation Helps Select Better $f$:**

1. Provides a **more reliable estimate** of generalization performance on unseen data.
2. Allows for **fair comparison** of different models or hyperparameters (e.g., regularization terms).
3. Helps select the best model $f^*$ that minimizes the cross-validation loss:

   $$
   f^* = \arg\min_{f \in \mathcal{H}} \; \hat{L}_{\text{cv}}(f)
   $$

You can find other cross validation techniques explanations here: 

  * [Top Cross-Validation Techniques with Python Code](https://www.analyticsvidhya.com/blog/2021/11/top-cross-validation-techniques-with-python-code/)

  * [Cross Validation Explained — Leave One Out, K Fold, Stratified, and Time Series Cross Validation Techniques](https://medium.com/@chanakapinfo/cross-validation-explained-leave-one-out-k-fold-stratified-and-time-series-cross-validation-0b59a16f2223)


## Bias-Variance Decomposition
Given a dataset $\mathcal{D} = \left\{ (x_i, y_i) \right\}_{i=1}^{k}$

Let,

$$
y = f(x)+\epsilon
$$

where $\epsilon$ is the noise in the dataset and it's independent from $x_i's$. Also we assume the noise $\epsilon$ is from a Gaussian distrubtion with mean zero and variance $\sigma^2$ (ie $\epsilon \sim \mathcal{N}(0, \sigma^2)$).

Given $\hat{f}$ then the True Risk is given as;

$$
\begin{align*}
\mathbb{E}\left[ (y - \hat{f}(x))^2\right] &= \mathbb{E}\left[ (f(x)+\epsilon - \hat{f}(x))^2\right] \quad \text{ since } y = f(x)+\epsilon\\
&=\mathbb{E}\left[ (f(x)-\hat{f}(x))^2+2(f(x)-\hat{f}(x))\epsilon + \epsilon^2\right]\\
&=\mathbb{E}\left[(f(x)-\hat{f}(x))^2\right] +2\mathbb{E}\left[(f(x)-\hat{f}(x))\right]\mathbb{E}\left[\epsilon\right] + \mathbb{E}\left[\epsilon^2\right] \quad \text{ but } \mathbb{E}\left[\epsilon^2\right] = \sigma^2 \text{ and } \mathbb{E}\left[\epsilon\right]=0\\
&=\mathbb{E}\left[(f(x)-\hat{f}(x))^2\right] + \sigma^2 
\end{align*}
$$

Now lets solve $\mathbb{E}\left[(f(x)-\hat{f}(x))^2\right]$. Let $\bar{f}$ be the average predictor which is given as  $\bar{f} = \mathbb{E}\left[\hat{f}(x)\right]$. Then we will have,

$$
\begin{align*}
 \mathbb{E}\left[(f(x)-\hat{f}(x))^2\right] &= \mathbb{E}\left[(f(x)-\bar{f}(x)+\bar{f}(x)-\hat{f}(x))^2\right]\\
 &= \mathbb{E}\left[(f(x)-\bar{f}(x))^2+2(f(x)-\bar{f}(x))(\bar{f}(x)-\hat{f}(x))+(\bar{f}(x)-\hat{f}(x))^2\right]\\
 &= \mathbb{E}\left[(f(x)-\bar{f}(x))^2\right] + 2\mathbb{E}\left[f(x)-\bar{f}(x)\right]\mathbb{E}\left[\bar{f}(x)-\hat{f}(x)\right]+\mathbb{E}\left[(\bar{f}(x)-\hat{f}(x))^2\right]\\
 &= \mathbb{E}\left[(f(x)-\bar{f}(x))^2\right] +\mathbb{E}\left[(\bar{f}(x)-\hat{f}(x))^2\right] \quad \text{ since } \mathbb{E}\left[\bar{f}(x)-\hat{f}(x)\right] = 0 \text{ as } \bar{f}(x) = \mathbb{E}\left[\hat{f}(x)\right]
\end{align*}
$$

Therefore our final equestion can be expressed as;

$$
\begin{align*}
 \mathbb{E}\left[ (y - \hat{f}(x))^2\right] &= \mathbb{E}\left[(f(x)-\bar{f}(x))^2\right] +\mathbb{E}\left[(\bar{f}(x)-\hat{f}(x))^2\right] + \sigma^2\\
 &= \text{Bias}^2 + \text{Variance} + \text{Noise}
\end{align*}
$$

where;

$$
\mathbb{E}_{\mathcal{D}, \epsilon}[(y - \hat{f}(x))^2] = \underbrace{\mathbb{E}_{\mathcal{D}}[(f(x) - \bar{f}(x))^2]}_{\text{Bias}^2} + \underbrace{\mathbb{E}_{\mathcal{D}}[(\hat{f}(x) - \bar{f}(x))^2]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Noise}}
$$


Thus the Bias Variance threshold is given as;

$$
\text{True Risk} = \text{Bias}^2 + \text{Variance} + \text{Noise}
$$

| High Bias                           | High Variance                       |
|-------------------------------------|-------------------------------------|
| - Underfitting (high training and test error)| - Overfitting (low training error, high test error)|
| - Model has not learned enough      | - Model is complex                  |
|  ** Train longer                    | - Small training data               |
| - Model is too simple               |  ** Feature selection. [1](https://www.analyticsvidhya.com/blog/2016/12/introduction-to-feature-selection-methods-with-an-example-or-how-to-select-the-right-variables/), [2](https://github.com/Younes-Charfaoui/Feature-Selection-Techniques), [3](https://medium.com/@debopamdeycse19/feature-selection-techniques-in-machine-learning-00a261e2574a)               |
|  ** Increase the features           |  ** Dimesionality Reduction         |
|  ** Change Hypothesis class         |  ** Data Augmentation               |


The above table compares high bias (underfitting) and high variance (overfitting). Entries marked with `**` indicate common **solutions** to address each issue.


## Maximum Likelihood Estimation
Here we are going to use the concept of MLE to find negative log-likelihood and also come up with some of the objective/loss functions.

Given a dataset 

$$
\begin{align*}
&\mathcal{D} = \left\{ (x_i, y_i) \right\}_{i=1}^{n}\\
&X = \left\{ x_i\right\}_{i=1}^{n}\\
&Y = \left\{ y_i\right\}_{i=1}^{n}\\
\end{align*}
$$

Let $h_{w} \in \mathcal{H}$ i.e a function $h$ with parameter $w$ in hypothesis space $\mathcal{H}$. If we assuming the outputs are drawn independently from a distribution $P(y_i | x_i; w)$,the **likelihood** of the data is:

$$
\begin{align*}
\mathcal{L}(w) &= P(Y | X; w)\\
               &= \prod_{i=1}^{n} P_i(y_i | x_i; w) \quad \text{ since they are independent}\\
               &= \prod_{i=1}^{n} P(y_i | x_i; w) \quad \text{ since they are identically ditributed}
\end{align*}
$$

Since the product of probabilities tends to zero or are very small (i.e between 0 and 1) we take its $log$ as $log$ is an increasing function and it changes the product of probabilities in to summation which simplifies computation to have **log-likelihood**:

$$
\begin{align*}
\log \mathcal{L}(w) &= \log \prod_{i=1}^{n} P(y_i | x_i; w)\\
                    &= \sum_{i=1}^{n} \log P(y_i | x_i; w)
\end{align*}
$$

Then, the **Maximum Likelihood Estimation (MLE)** objective is:

$$
\begin{align*}
w^* &= \arg\max_w \log \mathcal{L}(w)\\
    &= \arg\max_w \sum_{i=1}^{n} \log P(y_i | x_i; w)\\
    &= \arg\min_w \underbrace{- \sum_{i=1}^{n} \log P(y_i | x_i; w)}_{\text{negative log-likelihood (NLL)}}
\end{align*}
$$


Or equivalently, we minimize the **negative log-likelihood** (NLL):

$$
NLL(w) = -\sum_{i=1}^{k} \log P(y_i | x_i; w)
$$

## Examples: MLE Leading to Common Loss Functions

### 1. Mean Squared Error (MSE)

Given a dataset:

$$
\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n
$$

Assume that the target variable $y_i$ is generated as:

$$
y_i = w^\top x_i + \epsilon_i
$$

where $\epsilon_i \sim \mathcal{N}(0, \sigma^2)$.
So the conditional probability of $y_i$ given $x_i$ is:

$$
P(y_i \mid x_i; w) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - w^\top x_i)^2}{2\sigma^2} \right)
$$


Assuming i.i.d. data:

$$
\mathcal{L}(w) = \prod_{i=1}^{n} P(y_i \mid x_i; w) = \prod_{i=1}^{n} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - w^\top x_i)^2}{2\sigma^2} \right)
$$

Take the log:

$$
\begin{align*}
\log \mathcal{L}(w) &= \sum_{i=1}^{n} \log \left[ \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y_i - w^\top x_i)^2}{2\sigma^2} \right) \right]\\
                    &= \sum_{i=1}^{n} \left[ -\frac{1}{2} \log(2\pi\sigma^2) - \frac{(y_i - w^\top x_i)^2}{2\sigma^2} \right]
\end{align*}
$$

We now **minimize** the **negative log-likelihood**:

$$
\begin{align*}
w^*  &= \arg\min_w - \log \mathcal{L}(w)\\
     &= \arg\min_w - \sum_{i=1}^{n} \left[ -\frac{1}{2} \log(2\pi\sigma^2) - \frac{(y_i - w^\top x_i)^2}{2\sigma^2} \right]\\
     &= \arg\min_w \frac{n}{2} \log(2\pi\sigma^2) + \frac{1}{2\sigma^2}\sum_{i=1}^{n}(y_i - w^\top x_i)^2\\
     &= \arg\min_w \sum_{i=1}^{n}(y_i - w^\top x_i)^2 \quad \text{ since they dont depend on } w ,\frac{n}{2} \log(2\pi\sigma^2),\frac{1}{2\sigma^2}
\end{align*}
$$

Since $n\log(2\pi\sigma^2)$ and $\frac{1}{2\sigma^2}$ are constants (do not depend on $w$), minimizing $\mathcal{L}_{\text{NLL}}(w)$ is **equivalent** to minimizing:

$$
\sum_{i=1}^{n} (y_i - w^\top x_i)^2
$$

Thus, minimizing the **Mean Squared Error (MSE)**:

$$
\text{MSE}(w) = \frac{1}{n} \sum_{i=1}^{n} (y_i - w^\top x_i)^2
$$

is **equivalent** to **maximum likelihood estimation** under the assumption of **Gaussian noise** in the regression model.

### 2. Binary cross Entropy
Given a dataset:

$$
\mathcal{D} = \{(x_i, y_i)\}_{i=1}^n
$$

Assume that the target variable $y_i$ is generated from a Bernoulli distribution i.e

$$
P(y_i|x_i;w) = \hat{y_i}^{y_i}(1-\hat{y_i})^{1-y_i} \quad \text{ where } y_i \in \{0,1\}, 0\leq \hat{y_i}\leq 1
$$

Since $\hat{y_i}$ are probalities we can use sigmoid function to generate them;

$$
\hat{y_i} = \frac{1}{1+e^{w^\top x_i}}
$$

Assuming i.i.d. data:

$$
\mathcal{L}(w) = \prod_{i=1}^{n} P(y_i \mid x_i; w) = \prod_{i=1}^{n} \hat{y_i}^{y_i}(1-\hat{y_i})^{1-y_i}
$$

Take the log:

$$
\begin{align*}
\log \mathcal{L}(w) &= \sum_{i=1}^{n} \log \left[ \prod_{i=1}^{n} \hat{y_i}^{y_i}(1-\hat{y_i})^{1-y_i} \right]\\
                    &= \sum_{i=1}^{n} \log \left[ \hat{y_i}^{y_i}(1-\hat{y_i})^{1-y_i} \right]\\
                    &= \sum_{i=1}^{n} \left[ y_i\log \hat{y_i} +(1-y_i)\log (1-\hat{y_i}) \right]
\end{align*}
$$

We now **minimize** the **negative log-likelihood**:

$$
\begin{align*}
w^*  &= \arg\min_w - \log \mathcal{L}(w)\\
     &= \arg\min_w - \sum_{i=1}^{n} \left[ y_i\log \hat{y_i} +(1-y_i)\log (1-\hat{y_i}) \right]
\end{align*}
$$

Thus the resulting loss function is called the Binary Cross-Entropy given as:

$$
BCE(w) = - \sum_{i=1}^{n} \left[ y_i\log \hat{y_i} +(1-y_i)\log (1-\hat{y_i}) \right]
$$

So minimizing binary cross-entropy is equivalent to maximum likelihood estimation under the assumption that labels are drawn from a Bernoulli distribution with probability given by $\hat{y} = \frac{1}{1+e^{w^\top x}}$

**NOTE:**

In **Maximum Likelihood Estimation (MLE)**, to derive the **negative log-likelihood**, we assume that we know the **distribution of the outputs** $y_i$ given the inputs $x_i$ and model parameters $w$; that is, we assume a probabilistic model of the form:

$$
y_i \sim P(y_i \mid x_i; w)
$$

where $w$ represents the model parameters.

Now, we introduce another concept known as **Maximum A Posteriori (MAP) Estimation**.

## Maximum A Posteriori (MAP) Estimation

Given a dataset:

$$
\begin{aligned}
\mathcal{D} &= \left\{ (x_i, y_i) \right\}_{i=1}^{n} \\
X &= \left\{ x_i \right\}_{i=1}^{n} \\
Y &= \left\{ y_i \right\}_{i=1}^{n}
\end{aligned}
$$

Let $h_w \in \mathcal{H}$, i.e., a function $h$ parameterized by $w$, in hypothesis space $\mathcal{H}$.

In Maximum A Posteriori (MAP) Estimation, we assume we have prior knowledge about the distribution of the parameters $w$. Using Bayes' Rule:

$$
P(w \mid \mathcal{D}) = \frac{P(\mathcal{D} \mid w) \cdot P(w)}{P(\mathcal{D})}
$$

Where:

* $P(w \mid \mathcal{D})$: **Posterior** — what MAP tries to maximize
* $P(\mathcal{D} \mid w)$: **Likelihood** — same as in MLE
* $P(w)$: **Prior** over parameters
* $P(\mathcal{D})$: **Marginal likelihood** (a constant with respect to $w$)


The MAP estimate maximizes the **posterior**:

$$
\begin{aligned}
w_{\text{MAP}} &= \arg\max_w P(w \mid \mathcal{D}) \\
               &= \arg\max_w \frac{P(\mathcal{D} \mid w) \cdot P(w)}{P(\mathcal{D})} \\
               &= \arg\max_w P(\mathcal{D} \mid w) \cdot P(w) \quad \text{(since \( P(\mathcal{D}) \) is constant)}
\end{aligned}
$$

Because probabilities are small and prone to numerical instability, we apply the **log function** (monotonic, so it preserves maxima):

$$
\begin{aligned}
w_{\text{MAP}} &= \arg\max_w \left[ \log P(\mathcal{D} \mid w) + \log P(w) \right] \\
               &= \arg\max_w \left[ \underbrace{\log P(\mathcal{D} \mid w)}_{\text{log-likelihood}} + \underbrace{\log P(w)}_{\text{log-prior}} \right]
\end{aligned}
$$


In practice, we often minimize the negative log-posterior:

$$
w_{\text{MAP}} = \arg\min_w \left[ -\log P(\mathcal{D} \mid w) - \log P(w) \right]
$$

* The first term is the **negative log-likelihood** (same as MLE)
* The second term is the **negative log-prior** — acts as a regularization term

MAP is like MLE plus regularization, where the regularization reflects our prior belief about the parameters.

### Example 1: MAP with a Gaussian Prior (L2 regularization)

Assume 

$$
w \sim \mathcal{N}(0,\sigma^2)
$$

Where $d$ is the number of features and $n$ is number of samples i.e $X \in \mathbb{R}^{n\times d}$ which implies that $w \in \mathbb{R}^d$.

Now our prior will be 

$$
\begin{align*}
P(w) &= \prod_{j=1}^{d} P(w_j)\\
     &= \prod_{j=1}^{d} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{w_j^2}{2\sigma^2} \right)
\end{align*}
$$

Now solving log-prior,

$$
\begin{align*}
\log P(w) &= \log \prod_{j=1}^{d} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{w_j^2}{2\sigma^2} \right)\\
          &= \sum_{j=1}^{d} \left[ \log \frac{1}{\sqrt{2\pi\sigma^2}}  -\frac{ w_j^2}{2\sigma^2} \right]\\
          &= d \log \frac{1}{\sqrt{2\pi\sigma^2}} - \frac{1}{2\sigma^2}\sum_{j=1}^{d} w_j^2\\
          &= d \log \frac{1}{\sqrt{2\pi\sigma^2}} - \frac{1}{2\sigma^2} \|w\|_2^2 
\end{align*}
$$

Therefore we will have;

$$
\begin{align*}
w_{\text{MAP}} &= \arg\min_w \left[ -\log P(\mathcal{D} \mid w) - \log P(w) \right]\\
               &= \arg\min_w \left[ -\log P(\mathcal{D} \mid w) -\left(d \log \frac{1}{\sqrt{2\pi\sigma^2}} - \frac{1}{2\sigma^2} \|w\|_2^2\right) \right]\\
               &=\arg\min_w \left[ -\log P(\mathcal{D} \mid w) -d \log \frac{1}{\sqrt{2\pi\sigma^2}} + \frac{1}{2\sigma^2} \|w\|_2^2 \right]\\
               &=\arg\min_w \left[ -\log P(\mathcal{D} \mid w) + \frac{1}{2\sigma^2} \|w\|_2^2 \right]\\
               &=\arg\min_w \left[ -\log P(\mathcal{D} \mid w) + \lambda \|w\|_2^2 \right] \quad \text{ where } \lambda = \frac{1}{2\sigma^2} 
\end{align*}
$$


This shows that a Gaussian prior on $w$ leads to L2 regularization in the MAP objective which is the same as Ridge Regression.

### Example 2: MAP with a Laplace Prior (L1 regularization)

Assume the prior on each weight $w_j$ is a Laplace distribution (double exponential distribution):

$$
w_j \sim \text{Laplace}(0, b)
$$

Which implies

$$
P(w_j) = \frac{1}{2b} \exp\left(-\frac{|w_j|}{b}\right)
$$

Since weights are assumed to be iid, then the prior is:

$$
P(w) = \prod_{j=1}^{d} \frac{1}{2b} \exp\left(-\frac{|w_j|}{b}\right)
$$

Now computing the log-prior:

$$
\begin{align*}
\log P(w) &= \log \prod_{j=1}^{d} \frac{1}{2b} \exp\left(-\frac{|w_j|}{b}\right) \\
          &= \sum_{j=1}^{d} \left[ \log \frac{1}{2b} - \frac{|w_j|}{b} \right] \\
          &= d \log \frac{1}{2b} - \frac{1}{b} \sum_{j=1}^{d} |w_j| \\
          &= d \log \frac{1}{2b} - \frac{1}{b} \|w\|_1
\end{align*}
$$

Apply Bayes’ rule to get the MAP estimate:

$$
\begin{align*}
w_{\text{MAP}} &= \arg\min_w \left[ -\log P(\mathcal{D} \mid w) - \log P(w) \right] \\
               &= \arg\min_w \left[ -\log P(\mathcal{D} \mid w) - d \log \frac{1}{2b} + \frac{1}{b} \|w\|_1 \right] \\
               &= \arg\min_w \left[ -\log P(\mathcal{D} \mid w) + \frac{1}{b} \|w\|_1 \right] \\
               &= \arg\min_w \left[ -\log P(\mathcal{D} \mid w) + \lambda \|w\|_1 \right] \quad \text{ where } \lambda = \frac{1}{b}
\end{align*}
$$

This shows that a Laplace prior on $w$ leads to L1 regularization in the MAP objective which is the same as Lasso Regression.

## Gradient Descent from First-Order Taylor Approximation

The gradient of a function $f$ at a point $w$ yields the first-order Taylor approximation of $f$ around $w$, given as:

$$
f(u) \approx f(w) + \langle u - w, \nabla f(w) \rangle
$$

When $f$ is a convex function, this approximation is a lower bound of $f$. That is:

$$
f(u) \geq f(w) + \langle u - w, \nabla f(w) \rangle
$$

Therefore, for $w$ close to $w^{(t)}$, we can approximate $f(w)$ as:

$$
f(w) \approx f(w^{(t)}) + \langle w - w^{(t)}, \nabla f(w^{(t)}) \rangle
$$

We note that the above approximation might become loose if $w$ is far away from $w^{(t)}$. Therefore, we minimize jointly the distance between $w$ and $w^{(t)}$ and the approximation of $f$ around $w^{(t)}$. We also introduce a parameter $\eta$ which controls the trade-off between the two terms. This leads to the update rule as:

$$
w^{(t+1)} = \arg\min_w \left\{ \frac{1}{2} \| w - w^{(t)} \|^2 + \eta \left[ f(w^{(t)}) + \langle w - w^{(t)}, \nabla f(w^{(t)}) \rangle \right] \right\}
$$

Solving the optimization problem by taking the derivative with respect to $w$ and setting it to zero yields the gradient descent update rule.

Taking the objective function as $J(w)$:

$$
J(w) = \frac{1}{2} \| w - w^{(t)} \|^2 + \eta \left[ f(w^{(t)}) + \langle w - w^{(t)}, \nabla f(w^{(t)}) \rangle \right]
$$

Now solving for $\nabla_w J(w)$

$$
\begin{align*}
   \nabla_w J(w) &= \nabla_w \left[ \frac{1}{2} \| w - w^{(t)} \|^2 + \eta \left[ f(w^{(t)}) + \langle w - w^{(t)}, \nabla f(w^{(t)}) \rangle \right] \right]\\
    &= \nabla_w \left[ \frac{1}{2} (w - w^{(t)})^\top(w - w^{(t)}) + \eta f(w^{(t)}) + \eta (w - w^{(t)}) ^\top \nabla f(w^{(t)}) \right]\\
    &= \nabla_w \left[ \frac{1}{2} (w^\top w-w^\top w^{(t)} -(w^{(t)})^\top w +(w^{(t)})^\top (w^{(t)})  )  + \eta f(w^{(t)}) + \eta( w^\top \nabla f(w^{(t)}) - (w^{(t)})^\top \nabla f(w^{(t)}) )  \right]\\
    &=  \frac{1}{2} (2w -  2w^{(t)}) + \eta \nabla f(w^{(t)})\\
    &=  (w -  w^{(t)}) + \eta \nabla f(w^{(t)})
\end{align*}
$$

Next we set the gradient to zero i.e $\nabla_w J(w) = 0$

$$
\begin{align*}
   & (w -  w^{(t)}) + \eta \nabla f(w^{(t)}) = 0\\
   &w =  w^{(t)} -  \eta \nabla f(w^{(t)})
\end{align*}
$$

Therefor the gradient descent update rule is given as;

$$
\boxed{w^{(t+1)} = w^{(t)} - \eta \nabla f(w^{(t)})}
$$

## Gradient Descent (GD) Convergence

We want to show that GD converges, i.e.,

$$
f(\bar{w}) - f(w^*) \leq \varepsilon
$$

after a sufficient number of steps $T$, where:

* $f$ is convex and $\rho$-Lipschitz
* $\bar{w} = \frac{1}{T} \sum_{t=1}^T w^{(t)}$
* $w^*$ is the optimal point and $B$ be an upper bound on $\|w^*\|$ i.e $\|w^*\| \leq B$
* $w^{(t+1)} = w^{(t)} - \eta \nabla f(w^{(t)})$

**Proof:**

From the definition of $\bar{w}$ and using Jensen’s inequality we have;

$$
f(\bar{w}) - f(w^*) \leq \frac{1}{T} \sum_{t=1}^T \left[ f(w^{(t)}) - f(w^*) \right]
$$

Due to convexity of $f$ we have:

$$
f(w^{(t)}) - f(w^*) \leq \langle w^{(t)} - w^*, \nabla f(w^{(t)}) \rangle
$$

So combining the above two inqualies we obtain:

$$
f(\bar{w}) - f(w^*) \leq \frac{1}{T} \sum_{t=1}^T \langle w^{(t)} - w^*, \nabla f(w^{(t)}) \rangle
$$

To bound the right-hand side of the above inequality we use this lemma: Given the update rule $w^{(t+1)} = w^{(t)} - \eta \nabla f(w^{(t)})$, then it satisfies;

$$
\sum_{t=1}^T \langle w^{(t)} - w^*, \nabla f(w^{(t)}) \rangle \leq \frac{B^2}{2\eta} + \frac{\eta}{2} \sum_{t=1}^T \| \nabla f(w^{(t)}) \|^2
$$

If $f$ is $\rho$-Lipschitz, then $\| \nabla f(w^{(t)}) \| \leq \rho$. Therefore:

$$
\sum_{t=1}^T \langle w^{(t)} - w^*, \nabla f(w^{(t)}) \rangle \leq \frac{B^2}{2\eta} + \frac{\eta T \rho^2}{2}
$$

Now Choosing an optimal learning rate let say,

$$
\eta = \frac{B}{\rho \sqrt{T}}
$$

Then:

$$
\sum_{t=1}^T \langle w^{(t)} - w^*, \nabla f(w^{(t)}) \rangle \leq B \rho \sqrt{T}
$$

Dividing by $T$:

$$
f(\bar{w}) - f(w^*) \leq \frac{B \rho}{\sqrt{T}}
$$

If we run the GD algorithm on $f$ for $T$ steps with  $\eta = \frac{B}{\rho \sqrt{T}}$, then we will have,

$$
    f(\bar{w}) - f(w^*) \leq \frac{B \rho}{\sqrt{T}}
$$


To guarantee that the left-hand side is at most $\varepsilon$, we solve:

$$
\frac{B \rho}{\sqrt{T}} \leq \varepsilon \quad \Rightarrow \quad \sqrt{T} \geq \frac{B \rho}{\varepsilon} \quad \Rightarrow \quad T \geq \frac{B^2 \rho^2}{\varepsilon^2}
$$


Thus, Gradient Descent converges to within error $\varepsilon$ after:

$$
\boxed{T \geq \frac{B^2 \rho^2}{\varepsilon^2}}
$$

iterations, when minimizing a convex and $\rho$-Lipschitz function $f$, under the assumption that the optimal solution satisfies $\|w^*\| \leq B$.

## Stochastic Gradient Descent (SGD)

Stochastic Gradient Descent is an iterative optimization algorithm used to minimize a differentiable objective function. Unlike standard (batch) gradient descent, which computes the gradient using the entire dataset, SGD approximates the gradient using a single randomly chosen data point at each iteration.

From above, we have seen that standard gradient descent has the following update rule:

$$
w^{(t+1)} = w^{(t)} - \eta \nabla f(w^{(t)}) = w^{(t)} - \eta \left( \frac{1}{n} \sum_{i=1}^{n} \nabla f_i(w^{(t)}) \right)
$$

Here, $f(w) = \frac{1}{n} \sum_{i=1}^n f_i(w)$, where each $f_i(w)$ typically represents the loss on the $i$-th training example.

In Stochastic Gradient Descent, the full gradient $\nabla f(w^{(t)})$ is approximated by the gradient on a single example:

$$
w^{(t+1)} = w^{(t)} - \eta \nabla f_i(w^{(t)})
$$

where $i$ is chosen uniformly at random from $\{1, \dots, n\}$ at each iteration.

This stochastic estimate is unbiased but introduces variance, which affects convergence stability:

$$
\mathbb{E}_i [\nabla f_i(w)] = \nabla f(w)
$$

**Proof:**

Given that $i$ is selected uniformly at random from $\{1, 2, \dots, n\}$, the probability of selecting any particular index $i$ is $\frac{1}{n}$. Therefore, the expectation of the stochastic gradient is:

$$
\mathbb{E}_i[\nabla f_i(w)] = \sum_{i=1}^{n} \frac{1}{n} \nabla f_i(w) = \frac{1}{n} \sum_{i=1}^{n} \nabla f_i(w) = \nabla f(w)
$$

This confirms that the stochastic gradient estimator is unbiased.


## Mini-Batch Stochastic Gradient Descent

Mini-Batch SGD is an improvement between standard gradient descent and stochastic gradient descent (SGD). Instead of computing the gradient using the entire dataset or a single sample, it computes the gradient over a random subset (mini-batch) of data points at each iteration.

Given $B_t \subset \{1, 2, \dots, n\}$ be a randomly selected mini-batch of $m$ examples at iteration $t$. Then the update rule is given as:

$$
w^{(t+1)} = w^{(t)} - \eta \cdot \frac{1}{m} \sum_{i \in B_t} \nabla f_i(w^{(t)})
$$

Where:

* $m$: mini-batch size
* $B_t$: the mini-batch sampled uniformly without replacement

We can also show that the Mini-Batch SGD  gradient estimate is unbiased i.e

$$
\mathbb{E}_{B_t} \left[ g(w) \right] = \nabla f(w) \quad \text{ where } g(w) =  \frac{1}{m} \sum_{i \in B_t} \nabla f_i(w^{(t)})
$$

**Proof:**

We’re taking the expected value of the mini-batch gradient over all possible random mini-batches $B_t$ of size $m$. Since the indices are sampled uniformly, each data point $i$ has the same probability $\frac{m}{n}$ of being included in the mini-batch.

Using linearity of expectation:

$$
\mathbb{E}_{B_t}\left[ \frac{1}{m} \sum_{i \in B_t} \nabla f_i(w) \right] = \frac{1}{m} \sum_{i=1}^{n} \mathbb{P}(i \in B_t) \cdot \nabla f_i(w)
$$

Since:

$$
\mathbb{P}(i \in B_t) = \frac{m}{n}
$$

We substitute:

$$
\frac{1}{m} \sum_{i=1}^{n} \frac{m}{n} \nabla f_i(w) = \frac{1}{n} \sum_{i=1}^{n} \nabla f_i(w) = \nabla f(w)
$$

Therefore:

$$
\mathbb{E}_{B_t}[g(w)] = \nabla f(w)
$$

We can also note that Mini-Batch SGD  has a lower variance than the SGD due to averaging over multiple samples.

## Stochastic Gradient Descent (SGD) with Momentum

SGD with momentum is used to accelerate convergence and to smoothen updates, especially in directions of consistent gradients, by adding a "memory" of past gradients.

In SGD with Momentum we have the following update Rule:

We introduce a velocity vector $v^{(t)}$, which accumulates an exponentially decaying moving average of past gradients.

$$
\begin{aligned}
v^{(t+1)} &= \beta v^{(t)} + \nabla f_i(w^{(t)}) \\
w^{(t+1)} &= w^{(t)} - \eta v^{(t+1)}
\end{aligned}
$$

Where:

* $\beta \in [0,1)$ is the momentum coefficient,
* $v^{(0)} = 0$,
* $\nabla f_i(w^{(t)})$ is a stochastic gradient sampled at iteration $t$.

We can see that the velocity term is a weighted sum of previous gradients given as:

$$
v^{(t)} = \sum_{k=0}^{t-1} \beta^k \nabla f_{i_{t-k}}(w^{(t-k)})
$$

This shows that recent gradients have more weight (since $\beta^k$ decays with $k$).

## Neural Network

A neural network is a machine learning model inspired by the human brain that maps input data to output predictions by learning weights through layers of interconnected nodes (neurons). Neural networks are capable of modeling complex non-linear decision boundaries by composing multiple linear transformations and non-linear activation functions.

A feedforward neural network is made up of:

* An input layer that takes the features.
* One or more hidden layers that learn intermediate representations.
* An output layer that produces predictions.

Each layer applies a linear transformation followed by a non-linear activation function.

## Data Representation

Let:

* $n$: number of training examples
* $d$: number of features
* $X \in \mathbb{R}^{d \times n}$: input matrix
* $y \in \mathbb{R}^{1 \times n}$: target vector for regression or $y \in {0, 1}^{1\times n}$ for binary classification

For a 2-layer neural network (i.e., one hidden layer), we define:

* $W^{[1]} \in \mathbb{R}^{h \times d}$: weights for hidden layer
* $b^{[1]} \in \mathbb{R}^{h \times 1}$: bias for hidden layer
* $W^{[2]} \in \mathbb{R}^{1 \times h}$: weights for output layer
* $b^{[2]} \in \mathbb{R}$: bias for output layer

## Neural Network Architecture

Here we are going to use an example of a 2-layer feedforward neural network with:

*  Input layer ($d$ neurons)
*  Hidden layer ($h_1$ neurons)
*  Output layer ($h_2$ neurons)

Assuming our task is binary clasification and we use $\sigma$ (sigmoid) activation function on both layers.

Now let's see how our data and parameters are represented.

* **Data**: 

    **Features**: $X \in \mathbb{R}^{d\times n}$ which is the transpose of our original input data.
   
    **Targets**: $Y \in \mathbb{R}^{h_2\times n}$ which is the transpose of our original target data.

* **Layer 1**:

    **Weight**: $W_1 \in \mathbb{R}^{h_1\times d}$

    **Bias**: $b_1 \in \mathbb{R}^{h_1\times 1}$

* **Layer 2**:

    **Weight**: $W_2 \in \mathbb{R}^{h_2\times h_1}$

    **Bias**: $b_2 \in \mathbb{R}^{h_2\times 1}$

* **Loss Function**: Since this is a binary classification problem, we use binary cross entrpy.
$$
\mathcal{L} = - \frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
$$

### Forward Propagation
The Feedforward Propagation, also called Forward Pass, is the process consisting of computing all network nodes’ output values, starting with the first hidden layer until the last output layer, using at start either a subset or the entire dataset samples.

**Layer 1**

   * linear transformation 
$$
Z_1 = W_1X+b_1 \quad \text{where } Z_1 \in \mathbb{R}^{h_1\times n}
$$
   * non-linear transformation with $\sigma$ activation function
$$
A_1 = \sigma (Z_1) \quad \text{where } A_1 \in \mathbb{R}^{h_1\times n}
$$
   * Number of parameters in layer 1 is given as: $(h_1 \times d) + h_1 = h_1(d + 1)$.
 

**Layer 2**

   * linear transformation 

$$
Z_2 = W_2A_1+b_2 \quad \text{where } Z_2 \in \mathbb{R}^{h_2\times n}
$$

   * non-linear transformation with $\sigma$ activation function

$$
A_2 = \sigma (Z_2) \quad \text{where } A_2 \in \mathbb{R}^{h_2\times n}
$$

   * Number of parameters in layer 2 is given as: $(h_2 \times h_1) + h_2 = h_2(h_1+1)$.

**Output Layer**

   * Output

   $$
   A_2 = \hat{Y} \in \mathbb{R}^{h_2\times n}
   $$

## Loss Function

For binary classification, we use binary cross-entropy loss:

$$
\mathcal{L} = - \frac{1}{n} \sum_{i=1}^{n} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
$$

Matrix form:

$$
\mathcal{L} = - \frac{1}{n} \left[ Y \log(A_2) + (1 - Y) \log(1 - A_2) \right]
$$


 
### Backward Propagation
Backpropagation, or backward propagation of errors, is an algorithm working from the output nodes to the input nodes of a neural network using the chain rule to compute how much each activation unit contributed to the overall error.

Backpropagation automatically computes error gradients to then repeatedly adjust all weights and biases to reduce the overall error.

From our example our aim is to find 

$$
\displaystyle \frac{\partial L}{\partial A_2}, \quad \displaystyle \frac{\partial L}{\partial Z_2}, \quad \displaystyle \frac{\partial L}{\partial W_2}, \quad \displaystyle \frac{\partial L}{\partial b_2}, \quad \displaystyle \frac{\partial L}{\partial A_1}, \quad \displaystyle \frac{\partial L}{\partial Z_1}, \quad \displaystyle \frac{\partial L}{\partial W_1} \quad \text{ and } \displaystyle \frac{\partial L}{\partial b_1}
$$

Note that we are using our loss and binary cross entropy. But since we want to find its partial derivative w.r.t $A_2$ we will modify it to be

$$
L =  - \frac{1}{n} \left[ Y \log(A_2) + (1 - Y) \log(1 - A_2) \right]
$$

* $\displaystyle \frac{\partial L}{\partial A_2}$

$$
\displaystyle \frac{\partial L}{\partial A_2} =  \frac{1}{n} \left[\frac{A_2-Y}{A_2(1-A_2)}\right] \in \mathbb{R}^{h_2 \times n}
$$

* $\displaystyle \frac{\partial L}{\partial Z_2} = \displaystyle \frac{\partial L}{\partial A_2} \times \displaystyle \frac{\partial A_2}{\partial Z_2}$

$$
 \displaystyle \frac{\partial A_2}{\partial Z_2} = \sigma(Z_2)(1-\sigma(Z_2))= A_2(1-A_2) \in \mathbb{R}^{h_2 \times n} \quad \text{ since }\sigma(Z_2) = A_2\\
$$

Therefore we have,

$$
\begin{align*}
\displaystyle \frac{\partial L}{\partial Z_2} &= \displaystyle \frac{\partial L}{\partial A_2} \times \displaystyle \frac{\partial A_2}{\partial Z_2}\\
                                              &=  \frac{1}{n} \left[\frac{A_2-Y}{A_2(1-A_2)}\right] \times A_2(1-A_2)\\
                                              &= \frac{1}{n} \left[A_2-Y\right] \in \mathbb{R}^{h_2\times n}
\end{align*}
$$

* $\displaystyle \frac{\partial L}{\partial W_2} = \displaystyle \frac{\partial L}{\partial Z_2} \times \displaystyle \frac{\partial Z_2}{\partial W_2}$

$$
 \frac{\partial Z_2}{\partial W_2} = A_1^\top \in \mathbb{R}^{h_2\times h_1}
$$

Therefore we have,

$$
\displaystyle \frac{\partial L}{\partial W_2}  = \frac{1}{n} \left[A_2-Y\right]A_1^\top \in \mathbb{R}^{h_2 \times h_1}
$$

* $\displaystyle \frac{\partial L}{\partial b_2} = \displaystyle \frac{\partial L}{\partial Z_2} \times \displaystyle \frac{\partial Z_2}{\partial b_2}$

$$
 \frac{\partial Z_2}{\partial b_2} = I \quad \text{Identity}
$$

Therefore we have,

$$
\displaystyle \frac{\partial L}{\partial b_2}  = \frac{1}{n} \left[A_2-Y\right] \in \mathbb{R}^{h_2 \times n} \quad \text{ but } \displaystyle \frac{\partial L}{\partial b_2} \in \mathbb{R}^{h_2 \times 1} \quad \text{So, we will sum over the second dimension.}
$$

* $\displaystyle \frac{\partial L}{\partial A_1} = \displaystyle \frac{\partial L}{\partial Z_2} \times \displaystyle \frac{\partial Z_2}{\partial A_1}$

$$
 \displaystyle \frac{\partial Z_2}{\partial A_1} =  W_2^\top  \in \mathbb{R}^{h_1\times h_2}
$$

Therefore we have,

$$
\displaystyle \frac{\partial L}{\partial A_1} =  \frac{1}{n} \left[A_2-Y\right] W_2^\top \in \mathbb{R}^{h_1\times n}
$$

* $\displaystyle \frac{\partial L}{\partial Z_1} = \displaystyle \frac{\partial L}{\partial A_1} \times \displaystyle \frac{\partial A_1}{\partial Z_1}$

$$
 \displaystyle \frac{\partial A_1}{\partial Z_1} = \underbrace{\sigma(Z_1)(1-\sigma(Z_1))}_{\text{element-wise multip.}} \in \mathbb{R}^{h_1\times n}
$$

Therefore we have,

$$
\begin{align*}
\displaystyle \frac{\partial L}{\partial Z_1} &=  \frac{1}{n} \left[A_2-Y\right] W_2^\top (\sigma(Z_1)(1-\sigma(Z_1))) \quad \text{ Due to dimentionality we change to}\\
                                              &=   \frac{1}{n} \underbrace{\underbrace{W_2^\top \left[A_2-Y\right]}_{\text{matrix multip.}} (\sigma(Z_1)(1-\sigma(Z_1)))}_{\text{element-wise multip.}} \in \mathbb{R}^{h_1\times n}
\end{align*}
$$

* $\displaystyle \frac{\partial L}{\partial W_1} = \displaystyle \frac{\partial L}{\partial Z_1} \times \displaystyle \frac{\partial Z_1}{\partial W_1}$

$$
\displaystyle \frac{\partial Z_1}{\partial W_1} =  X^\top \in \mathbb{R}^{n\times d}
$$

Therefore we have,

$$
\displaystyle \frac{\partial L}{\partial W_1} = \frac{1}{n} W_2^\top \left[A_2-Y\right] (\sigma(Z_1)(1-\sigma(Z_1))) X^\top \in \mathbb{R}^{h_1\times d}
$$

* $\displaystyle \frac{\partial L}{\partial b_1} = \displaystyle \frac{\partial L}{\partial Z_1} \times \displaystyle \frac{\partial Z_1}{\partial b_1}$

$$
\displaystyle \frac{\partial Z_1}{\partial b_1} =  I \quad \text{Identity}
$$

Therefore we have,

$$
\displaystyle \frac{\partial L}{\partial b_1} = \frac{1}{n} W_2^\top \left[A_2-Y\right] (\sigma(Z_1)(1-\sigma(Z_1))) \in \mathbb{R}^{h_1\times n} \quad \text{So, we will sum over the second dimension to get } \mathbb{R}^{h_1\times 1}
$$



## Gradient Descent and Optimization

Using gradient descent, we update parameters in the direction that reduces the loss.

Let $\eta$ be the learning rate. The update rules:

* $W_1 \leftarrow W_1 - \eta \cdot \frac{\partial L}{\partial W_1}$
* $b_1 \leftarrow b_1 - \eta \cdot \frac{\partial L}{\partial b_1}$
* $W_2 \leftarrow W_2 - \eta \cdot \frac{\partial L}{\partial W_2}$
* $b_2 \leftarrow b_2 - \eta \cdot \frac{\partial L}{\partial b_2}$

Repeat this process for multiple epochs until the loss converges.


## **Vanishing and Exploding Gradient Problems in Deep Neural Networks**

In a deep neural network, at layer $\mathcal{l}$, we define the pre-activation and activation as follows:

$$
h^{\mathcal{l}} = \phi(Z^{\mathcal{l}}), \quad \text{where} \quad Z^{\mathcal{l}} = W^{\mathcal{l}} h^{\mathcal{l}-1}
$$

Here, $\phi$ is the activation function.

### **Gradient Behavior in Backpropagation**

During backpropagation, the gradient of the loss $L$ with respect to $Z^{\mathcal{l}}$ can be approximated as:

$$
\left\| \frac{\partial L}{\partial Z^{\mathcal{l}}} \right\| \approx \prod_{k = 1}^{\mathcal{l}} \|W^{\mathcal{k}}\| \cdot \|\phi'(Z^{\mathcal{k}-1})\|
$$

This product can either **explode** or **vanish**, depending on the magnitudes of the weights and activation derivatives.

### **Case 1: Exploding Gradients**

Occurs when:

$$
\|W^{\mathcal{k}}\| > 1
$$

Let

$$
C = \max_k \|W^{\mathcal{k}}\|
\Rightarrow \prod_{k} \|W^{\mathcal{k}}\|  \propto C^{\mathcal{l}}
$$

This exponential growth leads to **exploding gradients**, destabilizing training.

#### **Solutions:**

1. **Reduce depth** using residual/skip connections.
2. **Regularization** (e.g., $L_1$, $L_2$, or spectral norm regularization).
3. **Gradient clipping** to limit gradient magnitude.
4. **Normalization techniques**, such as BatchNorm or LayerNorm.

### **Case 2: Vanishing Gradients**

Occurs when:

$$
\|W^{\mathcal{k}}\| < 1 \quad \text{or} \quad \|\phi'(Z^{\mathcal{k}-1})\| < 1
$$

This leads to gradients approaching zero, making it difficult for earlier layers to learn.

#### **Solutions:**

1. **Reduce depth** via residual connections.
2. **Use non-saturating activation functions**:

   * Prefer **ReLU**, **Leaky ReLU**, **ELU**, or **Swish** over **sigmoid** or **tanh**.
3. **Proper weight initialization** (e.g., He or Xavier initialization).

### **Problem of Dying Neurons**

With ReLU, neurons can "die" (i.e., output zero for all inputs), especially when gradients become zero.

#### **Solutions:**

* Use **Leaky ReLU**, **PReLU**, or **ELU** to maintain non-zero gradients.


### **Depth and Skip Connections**

Depth refers to the number of layers or the length of the computational path from input to output. [**Skip connections**](https://arxiv.org/pdf/1512.03385v1) help by providing alternate shorter paths for gradient flow, effectively reducing the network's depth from a gradient propagation perspective.

### **Summary Table**

| Problem                | Solutions                                                                                                                       |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------- |
| **Vanishing Gradient** | - Use residual connections (reduce effective depth)  <br> - Use non-saturating activations <br> - Use proper initialization     |
| **Exploding Gradient** | - Use residual connections <br> - Regularization (e.g., spectral norm) <br> - Gradient clipping <br> - Normalization techniques |
| **Dying Neurons**      | - Use Leaky ReLU, ELU, or PReLU                                                                                                 |







 


     


