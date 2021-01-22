from data import *
from readdata import *
import numpy as np
import random
import math
from matplotlib import pyplot as plt

class Layer():
	def __init__(self, layer_size,input_size,activation='tanh',fixed_weights=False):
		self.fixed_weights = fixed_weights

		if self.fixed_weights == False:
			self.weights=[[random.random() for _ in range(input_size)] for _ in range(layer_size)]
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



class New_Model():
	def __init__(self,input_size = 2):
		self.input_size = input_size
		self.layers = []

	def add_layer(self,states=2,activation=None,fixed_weights=False):
		if len(self.layers)==0:
			self.layers.append(Layer(layer_size=states,input_size=self.input_size,activation=activation,fixed_weights=fixed_weights))
		else:
			self.layers.append(Layer(layer_size=states,input_size=len(self.layers[-1].states),activation=activation,fixed_weights=fixed_weights))

	def display(self):
		print("input: ",self.input_size)
		print("xxxxxxxxxxxxxxxxxxxx")
		for layer in self.layers:
			print(layer.states)
		print("-------------------------")
		for layer in self.layers:
			print(layer.weights)


	def __feed_forward(self):
		self.layers[0].feed_forward(self.input)

		for layer in range(1,len(self.layers)):
			self.layers[layer].feed_forward(self.layers[layer-1].states)

	def __stochastic_update(self,gradients):
		for layer in range(0,len(self.layers)):
			if self.layers[layer].fixed_weights == False:
				self.layers[layer].update(gradients[layer],learning_rate = self.learning_rate)

	def __back_propagation(self,target_label):
		delta = self.layers[-1].states[0] - target_label
		# print("delta--->",delta)
		gradients = [[] for _ in self.layers]
		for layer in reversed(range(0,len(self.layers))):

			

			if layer == len(self.layers)-1: #For the last layer
				delta = delta*self.layers[-1].activation_derivative(index = 0)


				for weight in range(0,len(self.layers[-1].weights)): ###This is for V_n's
					gradients[layer].append([])
					for w_index in range(0,len(self.layers[layer].weights[weight])):
						
						gradients[layer][-1].append(delta * self.layers[layer - 1].states[w_index])

			else:
				for weight_vector_index in range(0,len(self.layers[layer].weights))	:
					gradients[layer].append([])
					for w_index in range(0,len(self.layers[layer].weights[weight_vector_index])):

						gradients[layer][-1].append(delta * self.layers[layer+1].weights[0][0] * self.layers[layer].activation_derivative(index = weight_vector_index) * self.input[w_index])
		

		self.__stochastic_update(gradients)


	def __error(self,target_label): #this error is e^mu

		error = self.layers[-1].states[0]-target_label
		error = error*error/2
		self.error.append(error)

	def __validation(self,x_test,y_test):
		g_error = []
		for indx,x in enumerate(x_test):
			self.input = x
			self.__feed_forward()

			g_error.append(self.layers[-1].states[0] - y_test[indx])

			g_error[-1] = g_error[-1]*g_error[-1]


		return(sum(g_error)/len(g_error))




	def train(self, x_train,y_train,x_test,y_test,ephochs=10,learning_rate=0.05):
		self.learning_rate = learning_rate
		if x_train.shape[1]!=self.input_size:
			print("Data shape does not match model shape")
			return


		loss_ = []
		generalization_loss = []

		loss_.append(self.__validation(x_train,y_train))
		generalization_loss.append(self.__validation(x_test,y_test))

		print("\n\n=============================")
		print("----------------------------")
		print("INITIAL EMPIRICAL ERROR = ",loss_[-1])
		print("INITIAL VALIDATION ERROR = ",generalization_loss[-1])
		
		print("==========================")




		for epoch in range(0,ephochs):
			# self.display()
			print("\n\n=============================")
			print("Epoch "+str(epoch+1))
			print("=============================")

			self.error = []

			for i in range(0,x_train.shape[0]):
				# print("\nFeed forward Sample "+str(sample))

				sample = random.randint(0,x_train.shape[0]-1)


				
				self.input = list(x_train[sample])

				self.__feed_forward()

				self.__error(y_train[sample])

				# print("Error: ",self.error[sample])


				# self.display()

				self.__back_propagation(y_train[sample])



			
			loss_.append(sum(self.error)/len(self.error))

			generalization_loss.append(self.__validation(x_test,y_test))


			print("----------------------------")
			print("EPOCH EMPIRICAL ERROR = ",loss_[-1])
			print("EPOCH VALIDATION ERROR = ",generalization_loss[-1])
			
			print("==========================")
		




		plt.plot(loss_,marker='o',label="Empirical Error")
		plt.title("Errors")

		plt.plot(generalization_loss,marker='x',label="Validation Error")


		plt.xlabel('epoch')
		plt.ylabel('Loss')

		plt.legend()
		plt.show()

		



# data = Population(size=5,mean=0.5,variance=0.1,number_of_features=10)

# data.plot(data.dataset.shape[1])

P=200
Q=100

trainX,trainY,testX,testY = read_file("xi(1).csv","tau(1).csv",P,Q)



model = New_Model(input_size=len(trainX[0]))

model.add_layer(states = 10,activation = 'tanh',fixed_weights=False)
model.add_layer(states =1,activation = None,fixed_weights=1)

# model.display()

# print(trainX.shape)
model.train(trainX,trainY,testX,testY,ephochs=100,learning_rate=0.5)
