from Network import Model
from readdata import *
from matplotlib import pyplot as plt

"""
Experiment Description
----------------------
	Training on real-world dataset
"""


P=200
Q=52

learning_rate = 0.05
t_max = 30 #total epochs

trainX,trainY,testX,testY = read_file("real world dataset/xzscore.csv","real world dataset/tshift.csv",P,Q,shuffle=True)


# print(trainX.size)
model = Model(input_size=len(trainX[0]))

model.add_layer(states = 2,activation = 'tanh',fixed_weights=False)
model.add_layer(states =1,activation = None,fixed_weights=1/4)

E,E_test=model.train(trainX,trainY,testX,testY,ephochs=t_max,learning_rate=learning_rate,verbose=False)

# model.display()
plt.plot(E,label="Empirical Error(E)")
plt.title("Error-Time Graph")

plt.plot(E_test,label="Validation Error(E_test)")


plt.xlabel('Time')
plt.ylabel('Error')

plt.legend()

plt.savefig('Experiment8_fixed.png')
plt.show()


model = Model(input_size=len(trainX[0]))

model.add_layer(states = 2,activation = 'tanh',fixed_weights=False)
model.add_layer(states =1,activation = None,fixed_weights=False)

E,E_test=model.train(trainX,trainY,testX,testY,ephochs=t_max,learning_rate=learning_rate,verbose=False)

# model.display()
plt.plot(E,label="Empirical Error(E)")
plt.title("Error-Time Graph")

plt.plot(E_test,label="Validation Error(E_test)")


plt.xlabel('Time')
plt.ylabel('Error')

plt.legend()

plt.savefig('Experiment8_adaptive.png')
plt.show()






model = Model(input_size=len(trainX[0]))

model.add_layer(states = 5,activation = 'tanh',fixed_weights=False)
model.add_layer(states =1,activation = None,fixed_weights=1/4)

E,E_test=model.train(trainX,trainY,testX,testY,ephochs=t_max,learning_rate=learning_rate,verbose=False)

# model.display()
plt.plot(E,label="Empirical Error(E)")
plt.title("Error-Time Graph")

plt.plot(E_test,label="Validation Error(E_test)")


plt.xlabel('Time')
plt.ylabel('Error')

plt.legend()

plt.savefig('Experiment8_5_units.png')
plt.show()





