"""
Neural Network part of 
the NN project
Core element actually
"""

from sklearn.neural_network import MLPClassifier



def TrainAndValidate(X, y, X_validation, y_validation, outputLayerSize, display):
	"""
	-> first use of the MLPClassifier
	TODO:
		- a lot of things
	"""

	######################################
	# Neural Network Settings & Training #
	######################################
	clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
	                    hidden_layer_sizes=(3, outputLayerSize), random_state=1)
	clf.fit(X, y)


	##############
	# Prediction #
	##############
	score = 0
	prediction = clf.predict(X_validation)
	index = 0
	for element in prediction:
		if(display):
			print str(element) + " || " + y_validation[index]
		if(str(element) == str(y_validation[index])):
			score += 1
		index += 1

	score = (float(score) / float(len(y_validation)))*100
	if(display):
		print "[SCORE] "+str(float(score)) +"%"
	return score
