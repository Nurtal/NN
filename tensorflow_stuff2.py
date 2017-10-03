
##-----------------------##
## Simple Neural Network ##
##-----------------------##

import tensorflow as tf
import numpy as np






def generate_data(input_file):
	## Custom generation of data from
	## csv file to 2 numpy arrow
	## input_file : "DATA/train.csv" or "DATA/test.csv"

	batch_x = []
	batch_y = []
	label_to_position = {"Ctl":0, "RA":1, "SjS":2, "SLE":3, "SSc":4}

	input_data = open(input_file, "r")
	for line in input_data:
		line = line.replace("\n", "")
		line_in_array = line.split(",")
		label = line_in_array[0]

		## deal with data
		vector = line_in_array[1:]
		vector = np.asarray(vector)
		batch_x.append(vector)

		## deal with label
		vector_y = [0,0,0,0,0]
		vector_y[label_to_position[label]] = 1
		vector_y = np.asarray(vector_y)
		batch_y.append(vector_y)
	input_data.close()


	## return 2 2D numpy array
	batch_x = np.asarray(batch_x)
	batch_y = np.asarray(batch_y)

	return (batch_x, batch_y)










## get data
from tensorflow.examples.tutorials.mnist import input_data

# Python optimisation variables
learning_rate = 0.5
epochs = 10
total_batch = 161

# declare the training data placeholders
# input x - for 28 x 28 pixels = 784
x = tf.placeholder(tf.float32, [None, 88])
# now declare the output data placeholder - 10 digits
y = tf.placeholder(tf.float32, [None, 5])

# now declare the weights connecting the input to the hidden layer
W1 = tf.Variable(tf.random_normal([88, 44], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([44]), name='b1')

# and the weights connecting the hidden layer to the output layer
W2 = tf.Variable(tf.random_normal([44, 5], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([5]), name='b2')

# calculate the output of the hidden layer
hidden_out = tf.add(tf.matmul(x, W1), b1)
hidden_out = tf.nn.relu(hidden_out)

# now calculate the hidden layer output - in this case, let's use a softmax activated
# output layer
y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))

y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis=1))

# add an optimiser
optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# finally setup the initialisation operator
init_op = tf.global_variables_initializer()

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# start the session
with tf.Session() as sess:
	# initialise the variables
	sess.run(init_op)
	for epoch in range(epochs):
		avg_cost = 0
		for i in range(total_batch):
			batch_x, batch_y = generate_data("DATA/train.csv")
			_, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
			avg_cost += c / total_batch
		print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))

	test_x, test_y = generate_data("DATA/test.csv")
	print(sess.run(accuracy, feed_dict={x: test_x, y: test_y}))

















#### TUTORIAL SAVE ####
"""
## get data
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

# Python optimisation variables
learning_rate = 0.5
epochs = 10
batch_size = 100

# declare the training data placeholders
# input x - for 28 x 28 pixels = 784
x = tf.placeholder(tf.float32, [None, 784])
# now declare the output data placeholder - 10 digits
y = tf.placeholder(tf.float32, [None, 10])

# now declare the weights connecting the input to the hidden layer
W1 = tf.Variable(tf.random_normal([784, 300], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([300]), name='b1')

# and the weights connecting the hidden layer to the output layer
W2 = tf.Variable(tf.random_normal([300, 10], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([10]), name='b2')

# calculate the output of the hidden layer
hidden_out = tf.add(tf.matmul(x, W1), b1)
hidden_out = tf.nn.relu(hidden_out)

# now calculate the hidden layer output - in this case, let's use a softmax activated
# output layer
y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))

y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis=1))

# add an optimiser
optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

# finally setup the initialisation operator
init_op = tf.global_variables_initializer()

# define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


# start the session
with tf.Session() as sess:
	# initialise the variables
	sess.run(init_op)
	total_batch = int(len(mnist.train.labels) / batch_size)
	for epoch in range(epochs):
		avg_cost = 0
		for i in range(total_batch):
			batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
			_, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
			avg_cost += c / total_batch
		print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
	print(sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels}))
"""