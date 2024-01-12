---
title: Variational Autoencoders
status: hidden
date: 2022-01-11 00:00
category: Blog
slug: vae
tags: machine learning, generative ai
authors: Bishwa Karki
status: published
---

Training generative models and its efficiency and scalability was always a challenging problem. However VAE is a powerful framework for generative modeling, combining ideas from autoencoders and variational inference to efficiently learn probabilistic models with latent variables.

The vanilla autoencoder models each data were mapped to one point in the latent space. However, in VAE each data are mapped to a prior distribution(Gaussian, Bernoulli, etc). Doing so makes easier to create a new data points from the learned distribution and captures uncertainty in the data. This is a generative process which can generate new data points out of the distribution. However, sampling data points during training makes the process discontinuous and computing gradient is not possible. To solve this issue, they proposed a reparameterization trick which removes the randomness in sampling process and makes the operation differentiable which allows gradient to be calculated and optimization is possible.

To follow this probabilistic approach there are few changes made to the encoder and the loss function. And the decoder is kept similar to the vanilla autoencoder models. Here we will discuss from the probabilistic approaches.


## The Encoder
The encoder parameterizes the approximate posterior q(z|x). It takes the input x and output the distribution over z, i.e produces mean $\mu$ and standard deviation $\sigma$.

 q(z $\mid$ x) = $\mathcal{N}(z $\mid$ $\mu(x)$, $\sigma^2(x)$)


## The Decoder
The decoder is similar to the vanilla autoencoder where it generates samples from the latent variable z.
 p(x $\mid$ z) = Decoder(z)


## Training Objective
The training objective minimizes the ELBO, which is equivalent to minimizing the negative ELBO. The loss function consist of KL-divergence between the approximate posterior and the prior on the latent space and reconstruction loss.

\[ L(\theta, \phi; x) = -\mathbb{E}_{q_\phi(z \mid x)}[\log p_\theta(x \mid z)] + \text{KL}(q_\phi(z \mid x) \,||\, p(z)) \]