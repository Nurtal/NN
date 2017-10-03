"""
A few line of codes from te tensorflow
tutorial


-> Work under a specific conda environnement
-> python 3.5.4


=> TODO:
	- [DONE] split data into test and train
	- [DONE] read data from csv file
	- display graphe
	- run classifieur
	- interface with netabio
"""

import tensorflow as tf
import numpy as np
import tempfile
import urllib.request
import pandas as pd







def tuto_stuff():
	##---------------------##
	## Soft max regression ##
	##---------------------##

	## ADAPT TO REAL DATA

	## define the model
	x = tf.placeholder(tf.float32, [None, 88])
	W = tf.Variable(tf.zeros([88, 5])) # 88 pop for 5 disease
	b = tf.Variable(tf.zeros([5]))
	y = tf.nn.softmax(tf.matmul(x, W) + b)

	# Define loss and optimizer
	y_ = tf.placeholder(tf.float32, [None, 5])

	cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
	train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

	sess = tf.InteractiveSession()
	tf.global_variables_initializer().run()

	## train the model
	for _ in range(2000):

		## Get the data
		batch_xs, batch_ys = generate_data("DATA/train.csv")
		
		## train
		sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

	## Test trained model
	correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	## generate test data
	test_data, test_labels = generate_data("DATA/test.csv")

	## display results
	print(sess.run(accuracy, feed_dict={x: test_data, y_: test_labels}))





def conv_network_tuto():

	##-----------------------------------------##
	## Convolutionnal neural network from tuto ##
	##-----------------------------------------##

	from tensorflow.examples.tutorials.mnist import input_data
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

	x = tf.placeholder(tf.float32, shape=[None, 784])
	y_ = tf.placeholder(tf.float32, shape=[None, 10])

	def weight_variable(shape):
	  initial = tf.truncated_normal(shape, stddev=0.1)
	  return tf.Variable(initial)

	def bias_variable(shape):
	  initial = tf.constant(0.1, shape=shape)
	  return tf.Variable(initial)

	def conv2d(x, W):
	  return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

	def max_pool_2x2(x):
	  return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
	                        strides=[1, 2, 2, 1], padding='SAME')

	## First Convolutional layer
	W_conv1 = weight_variable([5, 5, 1, 32])
	b_conv1 = bias_variable([32])

	x_image = tf.reshape(x, [-1, 28, 28, 1])

	h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	## Second Convolutional Layer
	W_conv2 = weight_variable([5, 5, 32, 64])
	b_conv2 = bias_variable([64])

	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	## Densely Connected Layer
	W_fc1 = weight_variable([7 * 7 * 64, 1024])
	b_fc1 = bias_variable([1024])

	h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	## Dropout
	keep_prob = tf.placeholder(tf.float32)
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	## Readout Layer
	W_fc2 = weight_variable([1024, 10])
	b_fc2 = bias_variable([10])

	y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

	## Train and Evaluate the Model
	cross_entropy = tf.reduce_mean(
	    tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

	with tf.Session() as sess:
	  sess.run(tf.global_variables_initializer())
	  for i in range(2000):
	    batch = mnist.train.next_batch(50)
	    if i % 100 == 0:
	      train_accuracy = accuracy.eval(feed_dict={x: batch[0], y_: batch[1], keep_prob: 1.0})
	      print('step %d, training accuracy %g' % (i, train_accuracy))
	    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5})
	  print('test accuracy %g' % accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels, keep_prob: 1.0}))


def linear_model_tuto():


	train_file_name = "DATA/adult.data.txt"
	test_file_name = "DATA/adult.test.txt"


	CSV_COLUMNS = [
	    "age", "workclass", "fnlwgt", "education", "education_num",
	    "marital_status", "occupation", "relationship", "race", "gender",
	    "capital_gain", "capital_loss", "hours_per_week", "native_country",
	    "income_bracket"]
	df_train = pd.read_csv("DATA/adult.data.txt", names=CSV_COLUMNS, skipinitialspace=True)
	df_test = pd.read_csv("DATA/adult.test.txt", names=CSV_COLUMNS, skipinitialspace=True, skiprows=1)

	train_labels = (df_train["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)
	test_labels = (df_test["income_bracket"].apply(lambda x: ">50K" in x)).astype(int)

	def input_fn(data_file, num_epochs, shuffle):
	  """Input builder function."""
	  df_data = pd.read_csv(
	      tf.gfile.Open(data_file),
	      names=CSV_COLUMNS,
	      skipinitialspace=True,
	      engine="python",
	      skiprows=1)
	  
	  # remove NaN elements
	  df_data = df_data.dropna(how="any", axis=0)
	  labels = df_data["income_bracket"].apply(lambda x: ">50K" in x).astype(int)
	  return tf.estimator.inputs.pandas_input_fn(
	      x=df_data,
	      y=labels,
	      batch_size=100,
	      num_epochs=num_epochs,
	      shuffle=shuffle,
	      num_threads=5)



	## Base Categorical Feature Columns
	gender = tf.feature_column.categorical_column_with_vocabulary_list("gender", ["Female", "Male"])
	occupation = tf.feature_column.categorical_column_with_hash_bucket("occupation", hash_bucket_size=1000)
	education = tf.feature_column.categorical_column_with_vocabulary_list("education", ["Bachelors", "HS-grad", "11th", "Masters", "9th", "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th", "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th", "Preschool", "12th"])
	marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
	    "marital_status", [
	        "Married-civ-spouse", "Divorced", "Married-spouse-absent",
	        "Never-married", "Separated", "Married-AF-spouse", "Widowed"
	    ])
	relationship = tf.feature_column.categorical_column_with_vocabulary_list(
	    "relationship", [
	        "Husband", "Not-in-family", "Wife", "Own-child", "Unmarried",
	        "Other-relative"
	    ])
	workclass = tf.feature_column.categorical_column_with_vocabulary_list(
	    "workclass", [
	        "Self-emp-not-inc", "Private", "State-gov", "Federal-gov",
	        "Local-gov", "?", "Self-emp-inc", "Without-pay", "Never-worked"
	    ])
	native_country = tf.feature_column.categorical_column_with_hash_bucket(
	    "native_country", hash_bucket_size=1000)



	## Base Continuous Feature Columns
	age = tf.feature_column.numeric_column("age")
	education_num = tf.feature_column.numeric_column("education_num")
	capital_gain = tf.feature_column.numeric_column("capital_gain")
	capital_loss = tf.feature_column.numeric_column("capital_loss")
	hours_per_week = tf.feature_column.numeric_column("hours_per_week")


	age_buckets = tf.feature_column.bucketized_column(
	    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])

	## Defining The Logistic Regression Model
	base_columns = [
	    gender, native_country, education, occupation, workclass, relationship,
	    age_buckets,
	]
	crossed_columns = [
	    tf.feature_column.crossed_column(
	        ["education", "occupation"], hash_bucket_size=1000),
	    tf.feature_column.crossed_column(
	        [age_buckets, "education", "occupation"], hash_bucket_size=1000),
	    tf.feature_column.crossed_column(
	        ["native_country", "occupation"], hash_bucket_size=1000)
	]

	model_dir = tempfile.mkdtemp()
	m = tf.estimator.LinearClassifier(
	    model_dir=model_dir, feature_columns=base_columns + crossed_columns)


	# set num_epochs to None to get infinite stream of data.
	m.train(input_fn=input_fn(train_file_name, num_epochs=None, shuffle=True),steps=200)
	results = m.evaluate(
	    input_fn=input_fn(test_file_name, num_epochs=1, shuffle=False),
	    steps=None)
	print("model directory = %s" % model_dir)
	for key in sorted(results):
	  print("%s: %s" % (key, results[key]))




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
	train_proportion = 0.8

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



### TEST SPACE ###
#tuto_stuff()
#conv_network_tuto()







##----------------------##
## Wide & Deep Learning ##
##----------------------##
## combinaison d'une regression lineaire et dun reseau de neuronnes
## TODO: Adapt to real data

## data file
train_file_name = "DATA/adult.data.txt"
test_file_name = "DATA/adult.test.txt"
train_steps = 200

## => Define Base Feature Columns
gender = tf.feature_column.categorical_column_with_vocabulary_list(
    "gender", ["Female", "Male"])
education = tf.feature_column.categorical_column_with_vocabulary_list(
    "education", [
        "Bachelors", "HS-grad", "11th", "Masters", "9th",
        "Some-college", "Assoc-acdm", "Assoc-voc", "7th-8th",
        "Doctorate", "Prof-school", "5th-6th", "10th", "1st-4th",
        "Preschool", "12th"
    ])
marital_status = tf.feature_column.categorical_column_with_vocabulary_list(
    "marital_status", [
        "Married-civ-spouse", "Divorced", "Married-spouse-absent",
        "Never-married", "Separated", "Married-AF-spouse", "Widowed"
    ])
relationship = tf.feature_column.categorical_column_with_vocabulary_list(
    "relationship", [
        "Husband", "Not-in-family", "Wife", "Own-child", "Unmarried",
        "Other-relative"
    ])
workclass = tf.feature_column.categorical_column_with_vocabulary_list(
    "workclass", [
        "Self-emp-not-inc", "Private", "State-gov", "Federal-gov",
        "Local-gov", "?", "Self-emp-inc", "Without-pay", "Never-worked"
    ])

# To show an example of hashing:
occupation = tf.feature_column.categorical_column_with_hash_bucket(
    "occupation", hash_bucket_size=1000)
native_country = tf.feature_column.categorical_column_with_hash_bucket(
    "native_country", hash_bucket_size=1000)

# Continuous base columns.
age = tf.feature_column.numeric_column("age")
education_num = tf.feature_column.numeric_column("education_num")
capital_gain = tf.feature_column.numeric_column("capital_gain")
capital_loss = tf.feature_column.numeric_column("capital_loss")
hours_per_week = tf.feature_column.numeric_column("hours_per_week")

# Transformations.
age_buckets = tf.feature_column.bucketized_column(
    age, boundaries=[18, 25, 30, 35, 40, 45, 50, 55, 60, 65])


## => The Wide Model: Linear Model with Crossed Feature Columns
base_columns = [
    gender, native_country, education, occupation, workclass, relationship,
    age_buckets,
]

crossed_columns = [
    tf.feature_column.crossed_column(
        ["education", "occupation"], hash_bucket_size=1000),
    tf.feature_column.crossed_column(
        [age_buckets, "education", "occupation"], hash_bucket_size=1000),
    tf.feature_column.crossed_column(
        ["native_country", "occupation"], hash_bucket_size=1000)
]

## => The Deep Model: Neural Network with Embeddings
deep_columns = [
    tf.feature_column.indicator_column(workclass),
    tf.feature_column.indicator_column(education),
    tf.feature_column.indicator_column(gender),
    tf.feature_column.indicator_column(relationship),
    # To show an example of embedding
    tf.feature_column.embedding_column(native_country, dimension=8),
    tf.feature_column.embedding_column(occupation, dimension=8),
    age,
    education_num,
    capital_gain,
    capital_loss,
    hours_per_week,
]

## => Combining Wide and Deep Models into One
import tempfile
model_dir = tempfile.mkdtemp()
m = tf.estimator.DNNLinearCombinedClassifier(
    model_dir=model_dir,
    linear_feature_columns=crossed_columns,
    dnn_feature_columns=deep_columns,
    dnn_hidden_units=[100, 50])


## => Training and Evaluating The Model
# Define the column names for the data sets.
CSV_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race", "gender",
    "capital_gain", "capital_loss", "hours_per_week", "native_country",
    "income_bracket"
]

def maybe_download(train_data, test_data):
  """Maybe downloads training data and returns train and test file names."""
  if train_data:
    train_file_name = train_data
  else:
    train_file = tempfile.NamedTemporaryFile(delete=False)
    urllib.request.urlretrieve(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        train_file.name)  # pylint: disable=line-too-long
    train_file_name = train_file.name
    train_file.close()
    print("Training data is downloaded to %s" % train_file_name)

  if test_data:
    test_file_name = test_data
  else:
    test_file = tempfile.NamedTemporaryFile(delete=False)
    urllib.request.urlretrieve(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test",
        test_file.name)  # pylint: disable=line-too-long
    test_file_name = test_file.name
    test_file.close()
    print("Test data is downloaded to %s"% test_file_name)

  return train_file_name, test_file_name

def input_fn(data_file, num_epochs, shuffle):
  """Input builder function."""
  df_data = pd.read_csv(
      tf.gfile.Open(data_file),
      names=CSV_COLUMNS,
      skipinitialspace=True,
      engine="python",
      skiprows=1)
  # remove NaN elements
  df_data = df_data.dropna(how="any", axis=0)
  labels = df_data["income_bracket"].apply(lambda x: ">50K" in x).astype(int)
  return tf.estimator.inputs.pandas_input_fn(
      x=df_data,
      y=labels,
      batch_size=100,
      num_epochs=num_epochs,
      shuffle=shuffle,
      num_threads=5)


# set num_epochs to None to get infinite stream of data.
m.train(
    input_fn=input_fn(train_file_name, num_epochs=None, shuffle=True),
    steps=train_steps)
# set steps to None to run evaluation until all data consumed.
results = m.evaluate(
    input_fn=input_fn(test_file_name, num_epochs=1, shuffle=False),
    steps=None)
print("model directory = %s" % model_dir)
for key in sorted(results):
  print("%s: %s" % (key, results[key]))