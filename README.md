# adorad
The Tensor Library for Python, written in C++

Read: https://medium.com/@maciejkula/building-an-autodifferentiation-library-9ccf32c7a658
Motivation
There are a couple of reasons why I wanted to have my own autodiff/backprop framework, rather than use PyTorch or TensorFlow.
PyTorch and TF are quite slow when fitting models that require little computation per minibatch. In computer vision problems so much computation is done per minibatch that framework overhead is mostly a non-issue. This isn’t true of fitting matrix-factorization-style models, useful in the recommender systems community. Even on a GPU, fitting these models is very slow.
I want to be able to use my autodiff library to write and distribute models as Python packages with minimal dependencies. Being able to produce a fairly small and self-contained binary is an advantage over the rather heavy TF and PyTorch dependencies.
It was a fun learning experience, and allowed me to understand the inner workings of mature neural network libraries in a little bit more detail.