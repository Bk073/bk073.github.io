---
title: Batch Normalization
status: hidden
date: 2022-12-23 00:00
category: Blog
slug: batch-normalization
tags: machine learning, activation function
authors: Bishwa Karki
status: published
---


The main motivation for purposing Batch Normalization was to ease the training of deep learning models and help it converge faster. Deep Neural Networks are composed of multiple layers stacked on top of each others. During the forward propagating the input data is passed through all the layers and on doing so there is change in distribution of the inputs to each layer which is called as Internal Covariate shift. Lets visualize this mathematically with simple example:

## Relation with activation function

Before going onto the Batch Normalization, lets see some properties of some activation function. 

### Tanh

Here I will not be listing formula and graph of Tanh, as I guess most of us are familiar. 

```python
g = torch.Generator().manual_seed(2147483647)
x = torch.randn((500, 16), requires_grad=True, generator=g)
weights = torch.randn((500, 500), requires_grad=True, generator=g)

net = weights @ x 

non_linearity = torch.tanh(net)


plt.hist(non_linearity.view(-1).tolist(), 50);
```

This is a simple prototype of neural network with Tanh activation, and plot of the results after applying Tanh is depicted below:

<img src="/assets/img/blogs/tanh_distribution.png" width="100%" />

So we can see from the above histogram that Tanh takes most of the value to -1 and 1. And this is because of the reason that the input to the Tanh has a broad distribution and Tanh squash the exterme values to -1 and 1. Lets see the distribution of input to the Tanh:
```python
plt.hist(net.view(-1).tolist(), 50);
```

<img src="/assets/img/blogs/pre-active.png" width="100%" />

Now during back-propagation this nature of activation function will cause the problem. Lets see the gradient function of Tanh function used during back-propagation:

```python
grad_for_tan_h = += (1 - tanh**2) * previous_layer_grad
```

I wrote here **previous_layer_grad** because it's not important to show what I actually want. Here,

```python
if:
tanh = -1 or 1 
than grad_for_tan_h = 0
```

Now when we provide more and more samples to the network in the hope of getting our weights and bias better and better during training, but this doesn't happen because of the gradient of tanh being zero will destroy all the gardient of the network and make them inactivate.

Similar is the case with other activation function like **Sigmoid and ReLU** but this is not the case with **Leaky ReLU, Maxout and ELU**.

### Batch Normalization:

