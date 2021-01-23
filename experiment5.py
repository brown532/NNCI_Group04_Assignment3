from Network import Model
from readdata import *
from matplotlib import pyplot as plt
import numpy as np

"""
Experiment Description
----------------------
	Compare Different values of P
"""

R=10 #The average error on R runs of the training is taken

P_values=[20,50,200,500,1000,2000]
# Q=[int(p/2) for p in P]
Q=200

learning_rate = 0.05
t_max = 100 #total epochs


Errors_for_different_Ps =[]
Error_test_for_different_Ps = []

print(P_values)
for P in P_values:
	print("==================")
	print("iteration for P = "+str(P))

	trainX,trainY,testX,testY = read_file("xi(1).csv","tau(1).csv",P,Q)

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

	Errors_for_different_Ps.append(average_E)
	Error_test_for_different_Ps.append(average_Etest)


# print(Errors_for_different_Ps)


for index,error in enumerate(Errors_for_different_Ps):
	plt.plot(error,marker='.',label="P="+str(P_values[index]))#,label="Empirical Error(E)")


plt.title("Error-Time Graph")
plt.xlabel('Time')
plt.ylabel('Empirical Error (E)')

plt.legend()

plt.savefig('Experiment5_empirical_loss_for_different_P.png')
plt.show()



for index,error in enumerate(Error_test_for_different_Ps):
	plt.plot(error,marker='x',label="P="+str(P_values[index]))#,label="Empirical Error(E)")


plt.title("Error-Time Graph")
plt.xlabel('Time')
plt.ylabel('Validation Error (E_test)')

plt.legend()

plt.savefig('Experiment5_validation_loss_for_different_P.png')
plt.show()
