# Learning by Gradient Descent


## Abstract
Gradient Descent is a widely used optimization algorithm. It is however not without it's drawbacks as it can be computationally expensive. A commonly used variant is the Stochastic Gradient Descent (SGD). This project uses SGD to optimize the weight vectors of a shallow neural network for a regression task. It explores how the empirical error and validation error of the network changes in the course of training via SGD. It further examines the effects different learning rates have on the convergence of SGD training. Furthermore, it supports our machine learner's intuition that a larger dataset helps increase the generalization of a model by training the network with different sizes of training sets.


## How to run the experiments
- Make sure you have Python 3.7.6 installed
- `python -m pip install --upgrade pip`
- `pip install -r requirements.txt`
- `python Experiment*.py`


## Experiment Description
- **Experiment 1**: Observe Empirical error (E) with time, Observe Validation error (E_test) with time;	for one run of the training process

- **Experiment 2**: Compare E and E_test by taking the average over R runs

- **Experiment 4**: Return display the initial and final weights of the network

- **Experiment 5**: Compare Different values of P

- **Experiment 6**: Effects of different learning rates on empirical and validation error

- **Experiment 7**: Effects of using more hidden units (k>2) and adaptive hidden-to-output layer

- **Experiment 8**: Training on real-world dataset


## Authors
- [Brown Ogum](https://github.com/brown532)


