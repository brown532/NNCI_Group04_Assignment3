from data import *
import numpy as np
import random
import math


class Model():
	def __init__(self, hidden_units=2, input_size = 2):
		self.states=[[None]*input_size,[0]*hidden_units,[None]]
		self.weights = [  [ ],	[[1]*hidden_units]]

		for x in range(0,hidden_units):
			self.weights[0].append([random.random() for _ in range(input_size)])

		print(self.states)
		print(self.weights)



	def __feed_forward(self):
		for layer in range(1,len(self.states)):
			for state in range(0,len(self.states[layer])):

				if layer == len(self.states)-1:
					self.states[layer][state] = np.dot(self.states[layer-1],self.weights[layer-1][state])

				else:
					self.states[layer][state] = np.dot(self.states[layer-1],self.weights[layer-1][state])

					self.states[layer][state] = np.tanh(self.states[layer][state])


	def __error(self,target_label): #this error is e^mu
		error = self.states[-1][0]-target_label
		error = error*error/2
		self.error.append(error)


	def train(self, x_train,y_train,ephochs=10,learning_rate=0.05):
		if x_train.shape[1]!=len(self.states[0]):
			print("Data shape does not match model shape")
			return

		for epoch in range(0,ephochs):
			print("\n\n=============================")
			print("Epoch "+str(epoch))
			print("=============================")

			self.error = []

			for sample in range(0,x_train.shape[0]):
				print("\nFeed forward Sample "+str(sample))


				
				self.states[0] = list(x_train[sample])

				self.__feed_forward()

				self.__error(y_train[sample])

				print("Error: ",self.error[sample])


				print(self.states)
				print(self.weights)
		
		



data = Population(size=8,mean=0,variance=1.0,number_of_features=3)

model = Model(hidden_units=2,input_size=data.dataset.shape[1])

model.train(data.dataset,data.label,ephochs=1)

