
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def read_file(feature_path,label_path,P,Q):
	training_set=[]
	training_set_output = []

	test_set=[]

	if os.path.isfile(feature_path) and os.path.isfile(label_path):
		fin = open(label_path,'r')
		
		training_set_output = [float(x) for x in fin.readline().split(',')]


		fin.close()



		fin = open(feature_path,'r')


		
		for x in range(0,len(training_set_output)):
			training_set.append([])

		lines = []	
		for line in fin.readlines():
			line = [float(x) for x in line.split(',')]

			for y in range(0,len(line)):
				training_set[y].append(line[y])

		fin.close()

		test_set = training_set[P:Q+P]
		test_set_output = training_set_output[P:Q+P]

		training_set= training_set[0:P]
		training_set_output = training_set_output[0:P]

		return np.array(training_set),training_set_output,np.array(test_set),test_set_output
	else:
		if not os.path.isfile(label_path):
			print(label_path+" is not in working directory")
		else:
			print(feature_path+" is not in working directory")



def plot(dataset,labels):
		x_axis = list(range(1,dataset.shape[0]+1))

		ax = plt.gca()

		for x in range(0,dataset.shape[1]):
			y_axis = dataset[:,x]

			ax.scatter(x_axis,y_axis,marker='.',label="attribute "+str(x))#color=(random.random(),random.random(),random.random()))

		plt.xlabel('Sample',fontsize=14)
		# Set the y axis label of the current axis.
		plt.ylabel('Attribute Value',fontsize=14)
		# Set a title of the current axes.
		plt.title('Dataset',fontsize=20)#'First '+str(attributes_to_plot) + ' attributes of the dataset')
		# show a legend on the plot
		# plt.legend(loc='upper right',fontsize = 'x-small')
		# Display a figure.

		ax.xaxis.set_major_locator(MaxNLocator(nbins=30,integer=True))

		plt.show()


# trainX,trainY,testX,testY = read_file("xi(1).csv","tau(1).csv",30,2)

# plot(trainX,trainY)