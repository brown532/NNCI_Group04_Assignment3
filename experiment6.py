from Network import Model
from readdata import *
from matplotlib import pyplot as plt
import numpy as np

"""
Experiment Description
----------------------
	Effects of different learning rates on empirical and validation error
"""

R=10 #The average error on R runs of the training is taken

P=200
# Q=[int(p/2) for p in P]
Q=200

learning_rates = [0.05,0.1,0.2,0.4,0.8]
t_max = 100 #total epochs


Errors_for_different_LRs =[]
Error_test_for_different_LRs = []



trainX,trainY,testX,testY = read_file("xi(1).csv","tau(1).csv",P,Q)

print(learning_rates)

for learning_rate in learning_rates:
	print("==================")
	print("iteration for learning_rate = "+str(learning_rate))


	Es = []
	E_tests = []

	for y in range(0,R):
		print("\t Run "+str(y)+" of "+str(R))
		model = Model(input_size=len(trainX[0]))

		model.add_layer(states = 2,activation = 'tanh',fixed_weights=False)
		model.add_layer(states =1,activation = None,fixed_weights=1)

		E,E_test=model.train(trainX,trainY,testX,testY,ephochs=t_max,learning_rate=learning_rate)


		Es.append(E)
		E_tests.append(E_test)


	average_E = np.average(Es,axis=0)
	average_Etest = np.average(E_tests,axis = 0)

	Errors_for_different_LRs.append(average_E)
	Error_test_for_different_LRs.append(average_Etest)


# print(Errors_for_different_Ps)


for index,error in enumerate(Errors_for_different_LRs):
	plt.plot(error,label="Learning rate ="+str(learning_rates[index]))#,label="Empirical Error(E)")


plt.title("Error-Time Graph")
plt.xlabel('Time')
plt.ylabel('Average Empirical Error (E)')

plt.legend()

plt.savefig('Experiment6_empirical_loss_for_different_learning_rates.png')
plt.show()



for index,error in enumerate(Error_test_for_different_LRs):
	plt.plot(error,label="Learning rate ="+str(learning_rates[index]))#,label="Empirical Error(E)")


plt.title("Error-Time Graph")
plt.xlabel('Time')
plt.ylabel('Average Validation Error (E_test)')

plt.legend()

plt.savefig('Experiment6_validation_loss_for_different_learning_rates.png')
plt.show()





#######################
### Now using time-dependent learning rates
print("Using time-dependent learning rates")

########################
Errors_for_different_LRs =[]
Error_test_for_different_LRs = []
for learning_rate in learning_rates:
	print("==================")
	print("iteration for learning_rate = "+str(learning_rate))


	Es = []
	E_tests = []

	for y in range(0,R):
		print("\t Run "+str(y)+" of "+str(R))
		model = Model(input_size=len(trainX[0]))

		model.add_layer(states = 2,activation = 'tanh',fixed_weights=False)
		model.add_layer(states =1,activation = None,fixed_weights=1)

		E,E_test=model.train(trainX,trainY,testX,testY,ephochs=t_max,learning_rate=learning_rate,learning_rate_decay=0.01)


		Es.append(E)
		E_tests.append(E_test)


	average_E = np.average(Es,axis=0)
	average_Etest = np.average(E_tests,axis = 0)

	Errors_for_different_LRs.append(average_E)
	Error_test_for_different_LRs.append(average_Etest)


# print(Errors_for_different_Ps)


for index,error in enumerate(Errors_for_different_LRs):
	plt.plot(error,label="Initial learning rate ="+str(learning_rates[index]))#,label="Empirical Error(E)")


plt.title("Error-Time Graph")
plt.xlabel('Time')
plt.ylabel('Average Empirical Error (E)')

plt.legend()

plt.savefig('Experiment6_empirical_loss_for_different_time_dependent_learning_rates.png')
plt.show()



for index,error in enumerate(Error_test_for_different_LRs):
	plt.plot(error,label="Initial learning rate ="+str(learning_rates[index]))#,label="Empirical Error(E)")


plt.title("Error-Time Graph")
plt.xlabel('Time')
plt.ylabel('Average Validation Error (E_test)')

plt.legend()

plt.savefig('Experiment6_validation_loss_for_different_time_dependent_learning_rates.png')
plt.show()
