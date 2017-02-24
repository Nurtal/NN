"""
Data Manager for the NN projet
- Deal with data files, matrix creation & co ...
"""


def get_NumberOfPatients(PathToMatrixFile):
	"""
	-> Count number of patient in
	matrix file from RD
	-> return an int 
	"""
	inputDataFile = open(PathToMatrixFile, "r")
	numberOfPatient = 0
	for line in inputDataFile:
		numberOfPatient +=1
	inputDataFile.close()

	return int(numberOfPatient)



def generate_TrainingAndValidationMatrix(PathToMatrixFile, sizeOfValidationSet):
	"""
	-> Generate the training matrix
	-> Generate the validation matrix
	-> return both matrix in a tuple
	"""
	X = []
	X_validation = []
	numberOfPatient = get_NumberOfPatients(PathToMatrixFile)
	inputDataFile = open(PathToMatrixFile, "r")
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

	return(X, X_validation)


def generate_LabelsVectors(PathToMatrixLabelFile, binaryClassification, sizeOfValidationSet, numberOfPatient):
	"""
	-> Generate label vector for training data
	-> Generate label vector for validation data
	-> return both vectors in a tuple
	"""
	y = []
	y_validation = []
	inputLabelFile = open(PathToMatrixLabelFile, "r")
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

	return (y, y_validation)