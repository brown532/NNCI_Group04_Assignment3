import numpy as np
from numpy import linalg as LA
import random


class Layer():
	def __init__(self, layer_size,input_size,activation='tanh',fixed_weights=False):
		self.fixed_weights = fixed_weights

		if self.fixed_weights == False:
			self.weights=[[random.random() for _ in range(input_size)] for _ in range(layer_size)]

			for indx,weight in enumerate(self.weights):
				self.weights[indx] = weight/LA.norm(weight)

		else:
			self.weights=[[fixed_weights for _ in range(input_size)] for _ in range(layer_size)]

		self.states = [None]*layer_size
		self.activation = activation

	def feed_forward(self, input):
		self.states = [np.dot(input,weight) for weight in self.weights]

		if self.activation == 'tanh':
			self.states = np.tanh(self.states)


	def activation_derivative(self,index=0):
		if self.activation == 'tanh':
			return 1 - (self.states[index] *self.states[index])
		else:
			return 1

	def update(self,gradients,learning_rate = 0.05):
		for weight_vector_index in range(0,len(self.weights)):
			for weight_index in range(0,len(self.weights[weight_vector_index])):
				self.weights[weight_vector_index][weight_index] = self.weights[weight_vector_index][weight_index] - (learning_rate * gradients[weight_vector_index][weight_index])
