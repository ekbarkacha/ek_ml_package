# Logistic Regression

## Table of Contents
1. [Introduction](#introduction)
2. [Data Representation](#data-representation)
3. [Model Hypothesis](#model-hypothesis)
4. [Loss Function](#loss-function)
5. [Gradient Of Loss Function](#gradient)
6. [Method 2: When $y \in \{-1, 1\}$](#method-2-when--y-in--1-1-)
7. [Conclusion](#conclusion)

## Introduction

Logistic regression models a linear relationship between input features $X$ and a binary target vector $y$. It applies the sigmoid function to the linear combination of inputs to produce an output between 0 and 1, which is then thresholded to make a binary class prediction. Matrix operations enable efficient computation and scalability to large datasets.

---


## Data Representation

Let:

- $n$: number of training examples  
- $d$: number of features  

We define:

- $\mathbf{X} \in \mathbb{R}^{n \times d}$: feature matrix  
- $\mathbf{y} \in \{0,1\}^n$: target vector  
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

In logistic regression, we compute a linear combination of input features and model parameters, and then apply the **sigmoid function** to make the result into the range $[0, 1]$. The hypothesis function is:

$$
\hat{\mathbf{y}} = \sigma(\mathbf{X} \beta)
$$

where the sigmoid function $\sigma(z)$ is defined as:

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$
---

## Loss Function

To derive the loss function for logistic regression, we use maximum likelihood estimation (MLE), assuming that each target label is drawn from a Bernoulli distribution.

### Assumption: Bernoulli Targets

We assume that the training dataset $\{(x_i, y_i)\}_{i=1}^n$ are independent and identically distributed (iid), and that the target $y_{i} \in \{0, 1\}$ follows a Bernoulli distribution parameterized by $\theta_i$:

$$
y_{i} \sim \text{Bernoulli}(\theta_i)
$$

where $\theta_i = \hat{y}_{i} = \sigma(\beta^\top x_{i})$. The probability mass function of the Bernoulli distribution is:

$$
P(y_{i}\;\theta_i) = \theta_{i}^{y_{i}} (1 - \theta_{i})^{1 - y_{i}} \quad \text{ where } 0\le\theta_i \le 1, y_{i} = 0,1
$$

Now computing the likelihood $L(\beta)$:

$$
\begin{align*}
L(\beta) &= \prod_{i=1}^n P(y_{i}; \theta_i) \quad \text{ because of iid}\\ 
          &= \prod_{i=1}^n \theta_{i}^{y_{i}} (1 - \theta_{i})^{1 - y_{i}}\\
          &= \theta_{i}^{\sum_{i=1}^n y_{i} } (1 - \theta_{i})^{\sum_{i=1}^n (1 - y_{i})}
\end{align*}
$$

Next we computing the Log-Likelihood ($\log L(\beta)$),

$$
\begin{align*}
\log L(\beta) &=\log \left(\theta_{i}^{\sum_{i=1}^n y_{i} } (1 - \theta_{i})^{\sum_{i=1}^n (1 - y_{i})}\right)\\
               &= \sum_{i=1}^n y_{i} \log \theta_{i} + \sum_{i=1}^n (1 - y_{i}) \log (1 - \theta_{i})\\
               &= \sum_{i=1}^n  \left(y_{i} \log \theta_{i} + (1 - y_{i}) \log (1 - \theta_{i}) \right)
\end{align*}
$$

Finding the negative of Log-Likelihood $-\log L(\beta)$ we  have our final loss function as Negative-Log-Likelihood $NLL$ given as:

$$
NLL = - \sum_{i=1}^n \left( y_{i} \log \theta_{i} + (1 - y_{i}) \log (1 - \theta_{i})\right)
$$

Since $0\le\theta_{i} \le 1$ then we can use a sigmoid function to generate it hence we have $\theta_{i} = \hat{y}_{i} = \sigma(\beta^\top x_{i})$. And finally we have a loss function known as binary cross entropy given as 

$$
NLL = - \sum_{i=1}^n \left( y_{i} \log \sigma(z_{i}) + (1 - y_{i}) \log (1 - \sigma(z_{i}))\right) \quad \text{ where } \sigma(z_{i}) = \frac{1}{1 + e^{-z_{i}}}, z_{i} = \beta^\top x_{i}
$$

$$
\boxed{\text{binary cross entropy} = - \sum_{i=1}^n\left(y_{i} \log \sigma(z_{i}) + (1 - y_{i}) \log (1 - \sigma(z_{i}))\right)}
$$

---
## Gradient of the Loss Function

Now we want to find the gradient of our loss function Negative-Log-Likelihood $NLL$ using chain rule i.e.
$$
L = - \sum_{i=1}^n\left(y_{i} \log \sigma(z_{i}) + (1 - y_{i}) \log (1 - \sigma(z_{i}))\right)
$$
$$
\frac{d L}{d \beta} = \frac{d L}{dz_{i}} \times \frac{d z_{i}}{d \beta}
$$
### **NOTE**
$$
\sigma(z_{i}) = \frac{1}{1 + e^{-z_{i}}}, \quad \text{where } z_{i} = \beta^\top x_{i}
$$
To find the derivative of $\sigma(z_{i})$ with respect to $z_{i}$, we have:


$$
\boxed{
\begin{aligned}
\frac{d\sigma}{dz_{i}} &= \frac{d}{dz_{i}} \left( \frac{1}{1 + e^{-z_{i}}} \right) \\
&= \frac{d}{dz_{i}} \left( (1 + e^{-z_{i}})^{-1} \right) \\
&= -1 \cdot (1 + e^{-z_{i}})^{-2} \cdot \frac{d}{dz_{i}}(1 + e^{-z_{i}}) \\
&= - (1 + e^{-z_{i}})^{-2} \cdot (-e^{-z_{i}}) \\
&= \frac{e^{-z_{i}}}{(1 + e^{-z_{i}})^2} \\
\\
\text{Now recall:} \quad \sigma(z_{i}) &= \frac{1}{1 + e^{-z_{i}}} \quad \Rightarrow \quad 1 - \sigma(z_{i}) = \frac{e^{-z_{i}}}{1 + e^{-z_{i}}} \\
\\
\text{So:} \quad \frac{d\sigma}{dz_{i}} &= \sigma(z_{i})(1 - \sigma(z_{i}))
\end{aligned}
}
$$

Now continueing with the derivative of our binary cross entropy loss $L$ w.r.t $z_{i}$ we have

$$
\begin{aligned}
\frac{dL}{dz_{i}} &= -\left(y_{i}\cdot \frac{1}{\sigma(z_{i})} \cdot \sigma(z_{i})(1 - \sigma(z_{i}))  + (1 - y_{i}) \cdot \frac{1}{(1 - \sigma(z_{i}))} \cdot -\sigma(z_{i})(1 - \sigma(z_{i}))\right)\\
&= - \left[y_{i}(1 - \sigma(z_{i}))  - (1 - y_{i}) \sigma(z_{i})\right]\\
&= - \left[y_{i} - y_{i}\sigma(z_{i})  - \sigma(z_{i}) + y_{i}\sigma(z_{i})\right]\\
&= - \left[y_{i}  - \sigma(z_{i})\right]\\
&= \sigma(z_{i})-y_{i}
\end{aligned}
$$

Next we solve for $\frac{d z_{i}}{d \beta}$

$z_{i} = \beta^\top x_{i}$

$\frac{d z_{i}}{d \beta} = x_{i}$

Therefore we have our final gradient as

$$
\begin{align*}
\frac{d L}{d \beta} &= \frac{d L}{dz_{i}} \times \frac{d z_{i}}{d \beta}\\
                    &= \sum_{i=1}^n\left(\sigma(z_{i})-y_{i}\right) x_{i}
\end{align*}
$$

The gradient in vectorized form is:

$$
\boxed{ \frac{d L}{d \beta} = \sum_{i=1}^n \left( \sigma(z_i) - y_i \right) x_i }
$$

When $ X $ is a data matrix, the gradient becomes:

$$
\boxed{ \frac{d L}{d \beta} = X^\top \left( \sigma(X\beta) - y \right) }
$$

## Method 2: When $y \in \{-1, 1\}$

In some variants, target labels are represented using $-1$ and $+1$. The logistic regression model and loss function can be adjusted accordingly.

Now we are going to derive the hypothesis, loss function, and the gradient.

### Hypothesis
Let

$$
\mathcal{D} = \left\{ (x_i, y_i) \right\}_{i=1}^{n} \quad \text{where } x_i \in \mathbb{R}^{d},\quad y_i \in \{-1,1\},\quad w \in \mathbb{R}^{d}
$$

and $\hat{y} = w^\top x$

Assuming $P(Y=-1\mid X)$ and $P(Y=1\mid X)$ are defined, then

$$
\frac{P(Y=1\mid X)}{P(Y=-1\mid X)} \in (0,\infty)
$$

Now introducing the logarithm, we have

$$
\log \left(\frac{P(Y=1\mid X)}{P(Y=-1\mid X)} \right) = w^\top x \in \mathbb{R}
$$

This is equivalent to

$$
\begin{align*}
&\frac{P(Y=1\mid X)}{P(Y=-1\mid X)} = e^{w^\top x}\\
& P(Y=1\mid X) = \left[1-P(Y=1\mid X)\right]e^{w^\top x} \quad \text{since } P(Y=1\mid X) + P(Y=-1\mid X) =1\\
& P(Y=1\mid X)\left[1+e^{w^\top x}\right] = e^{w^\top x}\\
& P(Y=1\mid X) = \frac{e^{w^\top x}}{1+e^{w^\top x}} =  \frac{1}{1+e^{-w^\top x}}
\end{align*}
$$

Therefore we have

$$
P(Y=1\mid X) = \frac{1}{1+e^{-w^\top x}} = \sigma(w^\top x)
$$

And

$$
P(Y=-1\mid X) = \frac{1}{1+e^{w^\top x}} = \sigma(-w^\top x)
$$

Therefore, we can conclude our **Hypothesis** as

$$
\boxed{P(Y=y\mid X) = \frac{1}{1+e^{-yw^\top x}} = \sigma(yw^\top x)}
$$

where $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the sigmoid function.

### Loss Function

From **Maximum Likelihood Estimation (MLE)**, the negative log-likelihood is given as

$$
NLL(w) = -\sum_{i=1}^{n} \log P(y_i \mid x_i; w)
$$

From the hypothesis derived earlier, we have

$$
P(y_i \mid x_i; w) = \frac{1}{1 + e^{-y_i (w^\top x_i)}}
$$

Now solving for $NLL(w)$:

$$
\begin{align*}
NLL(w) &= -\sum_{i=1}^{n} \log \left(\frac{1}{1 + e^{-y_i (w^\top x_i)}}\right)\\
       &= -\sum_{i=1}^{n} \log \left(1 + e^{-y_i (w^\top x_i)}\right)^{-1}\\
       &= \sum_{i=1}^{n} \log \left(1 + e^{-y_i (w^\top x_i)}\right)
\end{align*}
$$

Therefore, our **Loss Function** is given as

$$
\boxed{\mathcal{L}(w) = \sum_{i=1}^{n} \log \left(1 + e^{-y_i (w^\top x_i)}\right)}
$$

### Gradient

Now we find the gradient of our loss function $\mathcal{L}(w)$ with respect to $w$:

$$
\begin{align*}
\frac{d \mathcal{L}(w)}{d w} &= \sum_{i=1}^{n} \frac{d}{d w} \log \left(1 + e^{-y_i (w^\top x_i)}\right) \\
&= \sum_{i=1}^{n} \frac{1}{1 + e^{-y_i (w^\top x_i)}} \cdot \left(-e^{-y_i (w^\top x_i)} \cdot y_i x_i\right) \\
&= -\sum_{i=1}^{n} \left( \frac{e^{-y_i (w^\top x_i)}}{1 + e^{-y_i (w^\top x_i)}} \right) y_i x_i \\
&= -\sum_{i=1}^{n} \left(1 - \frac{1}{1 + e^{-y_i (w^\top x_i)}} \right) y_i x_i \\
&= -\sum_{i=1}^{n} \left(1 - \sigma(y_i w^\top x_i)\right) y_i x_i
\end{align*}
$$

where $\sigma(z) = \frac{1}{1 + e^{-z}}$ is the sigmoid function.

Therefore our gradient $\nabla_w L$  is given as;
$$
\boxed{
\nabla_w L = -\sum_{i=1}^{n} \left(1 - \sigma(y_i w^\top x_i)\right) y_i x_i
}
$$



## Conclusion

Here's a table comparing the two formulations of logistic regression based on the label encoding: **$y \in {0, 1}$** vs. **$y \in {-1, 1}$**.


| Feature                       | **$y \in {0, 1}$**                                                                    | **$y \in {-1, 1}$**                                                         |
| ----------------------------- | --------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| **Label Encoding**            | 0 for negative class, 1 for positive class                                              | -1 for negative class, +1 for positive class                                  |
| **Hypothesis Function**       | $\hat{y} = \sigma(\beta^\top x)$                                                      | $P(y \mid x) = \sigma(y \cdot w^\top x)$                                    |
| **Sigmoid Argument**          | $\sigma(z) = \frac{1}{1 + e^{-\beta^\top x}}$                                         | $\sigma(y \cdot w^\top x)$                                                  |
| **Loss Function**             | $-\sum_{i=1}^n \left[y_i \log \sigma(z_i) + (1-y_i) \log(1-\sigma(z_i))\right]$ | $\sum_{i=1}^n \log(1 + e^{-y_i (w^\top x_i)})$                           |
| **Loss Name**                 | Binary Cross-Entropy                                                                    | Log-Loss (equivalent in nature, different form)                               |
| **Gradient of Loss Function** | $\nabla_\beta = X^\top(\sigma(X\beta) - y)$                                          | $\nabla_w = -\sum_{i=1}^n (1 - \sigma(y_i w^\top x_i)) y_i x_i$       |


Logistic regression models binary outcomes using a sigmoid function over a linear combination of features. We derived the binary cross-entropy loss using maximum likelihood and computed its gradient using the chain rule. The model can be expressed for both $y \in \{0, 1\}$ and $y \in \{-1, 1\}$, with slightly different forms of the loss function and gradient. The final vectorized gradient enables efficient optimization using gradient descent.
