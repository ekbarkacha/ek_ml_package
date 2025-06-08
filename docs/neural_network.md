# Neural Networks

## Table of Contents

1. [Introduction](#introduction)
2. [Data Representation](#data-representation)
3. [Neural Network Architecture](#neural-network-architecture)
4. [Forward Propagation](#forward-propagation)
5. [Loss Function](#loss-function)
6. [Backward Propagation](#backward-propagation)
7. [Gradient Descent and Optimization](#gradient-descent-and-optimization)
8. [Vanishing and Exploding Gradient](#vanishing-and-exploding-gradient) 
9. [Conclusion](#conclusion)

## Introduction

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

A neural network is made of interconnected neurons. Each of them is characterized by its **weight**, **bias** and **activation function**.

Here are other elements of this network.

**Input Layer**

The input layer takes raw input from the domain. No computation is performed at this layer. Nodes here just pass on the information (features) to the hidden layer. 

**Hidden Layer**

As the name suggests, the nodes of this layer are not exposed. They provide an abstraction to the neural network. 

The hidden layer performs all kinds of computation on the features entered through the input layer and transfers the result to the output layer.

**Output Layer**

It’s the final layer of the network that brings the information learned through the hidden layer and delivers the final value as a result. 


Here we are going to use an example of a 2-layer feedforward neural network with:

*  Input layer ($d$ neurons)
*  Hidden layer ($h_1$ neurons)
*  Output layer ($h_2$ neurons)

Assuming our task is binary clasification and we use $\sigma$ (sigmoid) activation function on both layers.

Now let's see how our data and parameters are represented.

* **Data**:

      - **Features**: $X \in \mathbb{R}^{d\times n}$ which is the transpose of our original input data.
      - **Targets**: $Y \in \mathbb{R}^{h_2\times n}$ which is the transpose of our original target data.

* **Layer 1**:

      - **Weight**: $W_1 \in \mathbb{R}^{h_1\times d}$
      - **Bias**: $b_1 \in \mathbb{R}^{h_1\times 1}$

* **Layer 2**:

      - **Weight**: $W_2 \in \mathbb{R}^{h_2\times h_1}$
      - **Bias**: $b_2 \in \mathbb{R}^{h_2\times 1}$

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

## Vanishing and Exploding Gradient

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


## Conclusion

Neural networks learn by combining linear transformations with non-linear activations, using forward and backward propagation to update their weights. We have explored a 2-layer network and detailed how gradients flow through the model during training.

We also discussed key training challenges which includes: **vanishing** and **exploding gradients** and how they can be addressed using techniques like **residual connections**, **proper initialization**, and **non-saturating activations**.

A solid understanding of these core concepts is essential for building deeper, more stable, and effective neural networks.
