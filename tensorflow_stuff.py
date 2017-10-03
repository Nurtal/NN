"""
A few line of codes from te tensorflow
tutorial


-> Work under a specific conda environnement
-> python 3.5.4


=> TODO:
	- split data into test and train
	- read data from csv file
	- display graphe
	- run classifieur
	- interface with netabio
"""

import tensorflow as tf



def tuto_stuff():
	##---------------------##
	## Soft max regression ##
	##---------------------##

	## ADAPT TO REAL DATA

	## TODO: import data

	## define the model
	x = tf.placeholder(tf.float32, [None, 88])
	W = tf.Variable(tf.zeros([88, 5])) # 88 pop for 5 disease
	b = tf.Variable(tf.zeros([5]))
	y = tf.nn.softmax(tf.matmul(x, W) + b)

	# Define loss and optimizer
	y_ = tf.placeholder(tf.float32, [None, 5])



	## FROM TUTORIAL
	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

	x = tf.placeholder(tf.float32, [None, 784])
	W = tf.Variable(tf.zeros([784, 10]))
	b = tf.Variable(tf.zeros([10]))

	## define the model
	y = tf.nn.softmax(tf.matmul(x, W) + b)

	y_ = tf.placeholder(tf.float32, [None, 10])

	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

	sess = tf.InteractiveSession()

	tf.global_variables_initializer().run()

	for _ in range(1000):
	  batch_xs, batch_ys = mnist.train.next_batch(100)

	  print(batch_xs[0][0].size)

	  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


	correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
	print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


##-----------------------##
## Linear Model Tutorial ##
##-----------------------##

"""
import tempfile
import urllib

train_file = tempfile.NamedTemporaryFile()
test_file = tempfile.NamedTemporaryFile()
urllib.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data", train_file.name)
urllib.urlretrieve("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test", test_file.name)
"""



### TRASH ###

def count_disease():

	disease = []
	data = open("DATA/newCytoData.csv", "r")
	for line in data:
		line = line.replace("\n", "")
		line_in_array = line.split(",")

		if(line_in_array[0] != "Disease" and line_in_array[0] not in disease):
			disease.append(line_in_array[0])
	data.close()

	print(disease)
	print(len(disease))


def split_data():
	## split a csv data file into
	## a train.csv and a test.csv file
	## TODO: adapt parameters to arguments

	## parameters
	label_to_count = {}
	label_to_train_proportion = {}
	label_to_patients_in_train_file = {}
	header = True
	train_proportion = 0.7

	## count the number of patients / label
	input_data = open("DATA/newCytoData.csv", "r")
	cmpt = 0
	for line in input_data:
		line = line.replace("\n", "")
		line_in_array = line.split(",")		
		if((header == True and cmpt != 0) or (header == False)):
			label = line_in_array[0]
			if label in label_to_count.keys():
				label_to_count[label] += 1
			else:
				label_to_count[label] = 1
		cmpt +=1
	input_data.close()

	## calculate the number of patient to
	## keep for each label
	label_to_train_proportion = label_to_count
	for lab in label_to_train_proportion.keys():
		number_of_train_patients = label_to_train_proportion[lab] * train_proportion
		label_to_train_proportion[lab] = number_of_train_patients
		label_to_patients_in_train_file[lab] = 0

	## split the data
	train_data = open("DATA/train.csv", "w")
	test_data = open("DATA/test.csv", "w")
	input_data = open("DATA/newCytoData.csv", "r")
	cmpt = 0
	for line in input_data:
		line = line.replace("\n", "")
		line_in_array = line.split(",")
		if((header == True and cmpt != 0) or (header == False)):
			label = line_in_array[0]
			if(float(label_to_patients_in_train_file[label]) <= float(label_to_train_proportion[label])):
				train_data.write(line+"\n")
				label_to_patients_in_train_file[label] += 1
			else:
				test_data.write(line+"\n")
		cmpt += 1
	input_data.close()
	test_data.close()
	train_data.close()



split_data()

