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



