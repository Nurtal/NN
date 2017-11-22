

## Rush for Seminar

from sklearn.neural_network import MLPClassifier



def run_nn(sizeOfValidationSet, hidden_layer_sizes):
	## rush stuff to get data for the seminar

	#############
	# Load data #
	#############

	# => Initialise settings
	X = []
	X_validation = []
	y = []
	y_validation = []
	numberOfPatient = "undef"
	#sizeOfValidationSet = 0.3
	#hidden_layer_sizes = 6
	binaryClassification = 1

	#input_data_file = "DATA/Luminex_custom_1.csv"
	input_data_file = "DATA/cyto_data_old_custom.csv"
	

	label_to_positions = {}



	## -> get number of patient
	## -> count the number of each different cases
	inputDataFile = open(input_data_file, "r")
	numberOfPatient = 0
	for line in inputDataFile:
		line = line.replace("\n", "")
		line_in_array = line.split(",")
		status = line_in_array[0]

		if numberOfPatient != 0:
			if status not in label_to_positions.keys():
				label_to_positions[status] = []
				label_to_positions[status].append(numberOfPatient)

			else:
				label_to_positions[status].append(numberOfPatient)

		numberOfPatient +=1
	inputDataFile.close()


	## -> Create training and validation matrix
	## Define the number of each case we want in the training and validation set
	label_to_quantity_in_validation_set = {}
	for label in label_to_positions.keys():
		label_to_quantity_in_validation_set[label] = float(float(sizeOfValidationSet)*float(len(label_to_positions[label])))

	## Create the validation set and training set
	label_to_catched_for_validation = {}
	for label in label_to_positions.keys():
		label_to_catched_for_validation[label] = 0

	inputDataFile = open(input_data_file, "r")
	train_file = open("DATA/Luminex_train.csv", "w")
	validation_file = open("DATA/Luminex_validation.csv", "w")

	cmpt = 0
	for line in inputDataFile:
		line = line.replace("\n", "")
		line_in_array = line.split(",")
		label = line_in_array[0]

		if(cmpt > 0):

			if(label_to_catched_for_validation[label] < label_to_quantity_in_validation_set[label]):
				validation_file.write(line+"\n")
				label_to_catched_for_validation[label] += 1

				## Fill data set
				vector = []
				index = 0
				for scalar in line_in_array:
					if(index > 0):
						vector.append(float(scalar))
					index += 1
				X_validation.append(vector)
				y_validation.append(line_in_array[0])

			else:
				train_file.write(line+"\n")

				vector = []
				index = 0
				for scalar in line_in_array:
					if(index > 0):
						vector.append(float(scalar))
					index += 1
				X.append(vector)
				y.append(line_in_array[0])

		cmpt += 1

	train_file.close()
	validation_file.close()
	inputDataFile.close()


	## Compute number of ouptut neuronne
	distinctValue = []
	for element in y:
		if element not in distinctValue:
			distinctValue.append(element)
	outputLayerSize = int(len(distinctValue))


	######################################
	# Neural Network Settings & Training #
	######################################
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
	                    hidden_layer_sizes=(hidden_layer_sizes, outputLayerSize), random_state=1)
	clf.fit(X, y)



	##############
	# Prediction #
	##############

	specific_score = {}
	good_prediction_for_label = {}
	number_of_case = {}

	score = 0
	prediction = clf.predict(X_validation)

	for element in prediction:
		good_prediction_for_label[element] = 0
		number_of_case[element] = 0
		specific_score[element] = 0

	index = 0
	for element in prediction:

		number_of_case[element] += 1

		print str(element) + " || " + y_validation[index]
		if(str(element) == str(y_validation[index])):
			score += 1
			good_prediction_for_label[element] += 1
		index += 1

	score = (float(score) / float(len(y_validation)))*100
	print "[SCORE] "+str(float(score)) +"%"


	for element in specific_score.keys():

		specific_score[element] = float(good_prediction_for_label[element]) / float(number_of_case[element])


	## log part
	log_file = open("rush_log.txt", "a")
	log_file.write("#--------------#\n")
	log_file.write("hidden layer sizes: "+str(hidden_layer_sizes)+"\n") 
	log_file.write("output layer size: "+str(outputLayerSize)+"\n")
	log_file.write("validation size set: "+str(sizeOfValidationSet)+"\n") 
	for element in specific_score:
		log_file.write(element +": "+str(specific_score[element])+" %\n") 
	log_file.write("Final score : "+str(score) +" %\n")
	log_file.close()





##------------##
## Global Run ##
##------------##

## init log file
log_file = open("rush_log.txt", "w")
log_file.close()
## Run the neural network
for x in xrange(1,300):
	hidden_layer_sizes = x
	for y in xrange(1, 9):
		sizeOfValidationSet = float(0.1*y)
		run_nn(sizeOfValidationSet, hidden_layer_sizes)


import matplotlib.pyplot as plt


def plot_log_file(log_file):
	## plot the precision of the nn
	## according to the values find in 
	## the log file

	specific_scores = {}
	global_score = []

	data = open(log_file, "r")

	for line in data:
		line = line.replace("\n", "")

		line_in_array = line.split(":")
		if(line_in_array[0] == "Final score "):
			score = line_in_array[1]
			score = score.replace(" ", "")
			score = score.replace("%", "")

			global_score.append(float(score))

	data.close()

	plt.plot(global_score)
	plt.show()



def generate_new_data_file():
	## custum generation of data file

	input_file = open("cyto_data_old.csv", "r")
	output_file = open("DATA/cyto_data_old_custom.csv", "w")

	for line in input_file:
		line = line.replace("\n", "")
		line_in_array = line.split(",")

		if(line_in_array[0] == "SjS" or line_in_array[0] == "SSc"):
			output_file.write(line+"\n")

	output_file.close()
	input_file.close()


#generate_new_data_file()
plot_log_file("rush_log.txt")