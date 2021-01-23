from Network import Model
from readdata import *
from matplotlib import pyplot as plt

"""
Experiment Description
----------------------
	Observe Empirical error (E) with time
	Observe Validation error (E_test) with time

	for one run of the training process
"""


P=200
Q=100

learning_rate = 0.05
t_max = 100 #total epochs

trainX,trainY,testX,testY = read_file("xi(1).csv","tau(1).csv",P,Q)



model = Model(input_size=len(trainX[0]))

model.add_layer(states = 2,activation = 'tanh',fixed_weights=False)
model.add_layer(states =1,activation = None,fixed_weights=1)

# model.display()

# print(trainX.shape)
E,E_test=model.train(trainX,trainY,testX,testY,ephochs=t_max,learning_rate=learning_rate)


plt.plot(E,marker='.',label="Empirical Error(E)")
plt.title("Error-Time Graph")

plt.plot(E_test,marker='x',label="Validation Error(E_test)")


plt.xlabel('Time')
plt.ylabel('Error')

plt.legend()

plt.savefig('Experiment1.png')
plt.show()
