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

### 1. Tanh

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