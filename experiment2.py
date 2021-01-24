from Network import Model
from readdata import *
from matplotlib import pyplot as plt
import numpy as np

"""
Experiment Description
----------------------
	Compare E and E_test by taking the average over R runs
"""

R=50

P=200
Q=100

learning_rate = 0.05
t_max = 100 #total epochs

trainX,trainY,testX,testY = read_file("xi(1).csv","tau(1).csv",P,Q)

Es = []
E_tests = []

for i in range(0,R):
	print("Run "+str(i+1)+" of "+str(R))
	model = Model(input_size=len(trainX[0]))

	model.add_layer(states = 2,activation = 'tanh',fixed_weights=False)
	model.add_layer(states =1,activation = None,fixed_weights=1)

	# model.display()

	# print(trainX.shape)
	E,E_test=model.train(trainX,trainY,testX,testY,ephochs=t_max,learning_rate=learning_rate)
	Es.append(E)
	E_tests.append(E_test)


average_E = np.average(Es,axis=0)
average_Etest = np.average(E_tests,axis = 0)


plt.plot(average_E,label="Empirical Error(E)")
plt.title("Error-Time Graph")

plt.plot(average_Etest,label="Validation Error(E_test)")


plt.xlabel('Time')
plt.ylabel('Averaged Error')

plt.legend()

plt.savefig('Experiment2.png')
plt.show()
