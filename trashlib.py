"""
trashlib, 
currently a functionnal script
"""


from sklearn.neural_network import MLPClassifier


#############
# Load data #
#############

# => Initialise settings
X = []
X_validation = []
y = []
y_validation = []
numberOfPatient = "undef"
sizeOfValidationSet = 30
binaryClassification = 1

# -> get number of patient
inputDataFile = open("../RD/DATA/MATRIX/discrete_processed_binary.csv", "r")
numberOfPatient = 0
for line in inputDataFile:
	numberOfPatient +=1
inputDataFile.close()

# -> test data size
if(sizeOfValidationSet >= numberOfPatient):
	print "[!] not enough patient for training and validation"

# => Create training and validation matrix
inputDataFile = open("../RD/DATA/MATRIX/discrete_processed_binary.csv", "r")
cmpt = 0
for line in inputDataFile:
	lineWithoutBackN = line.split("\n")
	lineWithoutBackN = lineWithoutBackN[0]
	lineInArray = lineWithoutBackN.split(";")
	vector = []
	for scalar in lineInArray:
		vector.append(int(scalar))

	if(cmpt < (numberOfPatient - sizeOfValidationSet)):
		X.append(vector)
	else:
		X_validation.append(vector)
	cmpt +=1
inputDataFile.close()

# => Create label from file
inputLabelFile = open("../RD/DATA/MATRIX/discrete_processed_binary_label.csv", "r")
cmpt = 0
for line in inputLabelFile:
	lineWithoutBackN = line.split("\n")
	lineWithoutBackN = lineWithoutBackN[0]

	scalar = lineWithoutBackN

	if(binaryClassification):
		if(lineWithoutBackN != "control"):
			scalar = "Malade"
		else:
			scalar = "Control"

	if(cmpt < (numberOfPatient - sizeOfValidationSet)):
		y.append(scalar)
	else:
		y_validation.append(scalar)
	cmpt += 1
inputLabelFile.close()

# -> compute number of ouptut neuronne
outputLayerSize = "undef"
if(binaryClassification):
	outputLayerSize = 2
else:
	distinctValue = []
	for element in y:
		if element not in distinctValue:
			distinctValue.append(element)
	outputLayerSize = int(len(distinctValue))


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
	print str(element) + " || " + y_validation[index]
	if(str(element) == str(y_validation[index])):
		score += 1
	index += 1

score = (float(score) / float(len(y_validation)))*100
print "[SCORE] "+str(float(score)) +"%"