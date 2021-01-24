from Network import Model
from readdata import *
from matplotlib import pyplot as plt
import numpy as np

"""
Experiment Description
----------------------
	Effects of using more hidden units (k>2) and adaptive hidden-to-output layer
"""

R=10

P=200
Q=100

learning_rate = 0.05
t_max = 100 #total epochs

trainX,trainY,testX,testY = read_file("xi(1).csv","tau(1).csv",P,Q)



k_values = [1,2,4]

empirical_error_for_k=[]
validation_error_for_k = []
print("k values = ",k_values)
for k in k_values:
	print("===============")
	print("Running for k = "+str(k))

	Es = []
	E_tests = []
	for i in range(0,R):
		print("\tRun "+str(i+1)+" of "+str(R))
		model = Model(input_size=len(trainX[0]))

		model.add_layer(states = k,activation = 'tanh',fixed_weights=False)
		model.add_layer(states =1,activation = None,fixed_weights=1)

		# model.display()

		# print(trainX.shape)
		E,E_test=model.train(trainX,trainY,testX,testY,ephochs=t_max,learning_rate=learning_rate)
		Es.append(E)
		E_tests.append(E_test)


	average_E = np.average(Es,axis=0)
	average_Etest = np.average(E_tests,axis = 0)

	empirical_error_for_k.append(average_E)
	validation_error_for_k.append(average_Etest)


for index,error_ in enumerate(empirical_error_for_k):
	plt.plot(error_,label="k="+str(k_values[index]))

plt.title("Empirical Error-Time Graph")
plt.xlabel('Time')
plt.ylabel('Averaged Error')

plt.legend()

plt.savefig('Experiment7_dffierent_K_empirical_error_FIXED_Vk.png')
# plt.show()
plt.close()

for index,error_ in enumerate(validation_error_for_k):
	plt.plot(error_,label="k="+str(k_values[index]))

plt.title("Validation Error-Time Graph")
plt.xlabel('Time')
plt.ylabel('Averaged Error')

plt.legend()

plt.savefig('Experiment7_dffierent_K_validation_error_FIXED_Vk.png')
# plt.show()
plt.close()



"""
	Second part: run the for different adaptive vk
"""

print("Now--- Running for adaptive hidden-to-output weights Vk")
empirical_error_for_k=[]
validation_error_for_k = []
print("k values = ",k_values)
for k in k_values:
	print("===============")
	print("Running for k = "+str(k))

	Es = []
	E_tests = []
	for i in range(0,R):
		print("\tRun "+str(i+1)+" of "+str(R))
		model = Model(input_size=len(trainX[0]))

		model.add_layer(states = k,activation = 'tanh',fixed_weights=False)
		model.add_layer(states =1,activation = None,fixed_weights=False)

		# model.display()

		# print(trainX.shape)
		E,E_test=model.train(trainX,trainY,testX,testY,ephochs=t_max,learning_rate=learning_rate)
		Es.append(E)
		E_tests.append(E_test)


	average_E = np.average(Es,axis=0)
	average_Etest = np.average(E_tests,axis = 0)

	empirical_error_for_k.append(average_E)
	validation_error_for_k.append(average_Etest)

for index,error_ in enumerate(empirical_error_for_k):
	plt.plot(error_,label="k="+str(k_values[index]))
plt.title("Empirical Error-Time Graph")
plt.xlabel('Time')
plt.ylabel('Averaged Error')

plt.legend()

plt.savefig('Experiment7_dffierent_K_empirical_error_ADAPTIVE_Vk.png')
# plt.show()
plt.close()

for index,error_ in enumerate(validation_error_for_k):
	plt.plot(error_,label="k="+str(k_values[index]))

plt.title("Validation Error-Time Graph")
plt.xlabel('Time')
plt.ylabel('Averaged Error')

plt.legend()

plt.savefig('Experiment7_dffierent_K_validation_error_ADAPTIVE_Vk.png')
# plt.show()
# plt.close()

