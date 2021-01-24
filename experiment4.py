from Network import Model
from readdata import *
from matplotlib import pyplot as plt

"""
Experiment Description
----------------------
	Return display the initial and final weights of the network
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
initial_weights,final_weights=model.train(trainX,trainY,testX,testY,ephochs=t_max,learning_rate=learning_rate,return_weights=True,verbose=False)

# print(initial_weights)
# print("\n==========================================\n")

# print(final_weights)

for w in range(0,initial_weights.shape[0]):
	fig, axs = plt.subplots(2)
	axs[0].bar(np.arange(initial_weights[w].size),initial_weights[w])
	axs[0].set_title('Initial w_' +str(w)+'weight vector')
	axs[0].set_ylabel('value')
	
	axs[1].bar(np.arange(final_weights[w].size),final_weights[w])
	axs[1].set_title('Final w_' +str(w)+' weight vector')
	axs[1].set_ylabel('value')
	axs[1].set_xlabel('feature')

	plt.savefig('Experiment4_weightvector'+str(w)+'.png')
	plt.show()

# plt.plot(E,marker='.',label="Empirical Error(E)")
# plt.title("Error-Time Graph")

# plt.plot(E_test,marker='x',label="Validation Error(E_test)")


# plt.xlabel('Time')
# plt.ylabel('Error')

# plt.legend()

# plt.savefig('Experiment1.png')
# plt.show()
