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
			if(lineWithoutBackN != "control" and lineWithoutBackN != "Control"):
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



def reformat_inputFile(input_file_name, index_file_name):
	"""
	-> reformat an input file, the input file have to contain only 
	   binary values (0 or 1) except for the patient identifiant.
	-> patient identifiant is used to retrieve the patient patient diagnostic
	   using the index file patient
	-> input_file_name is the name of the input file (generated by the binaryzation
	   function from RD)
	-> index_file_name is the name of the patient index file
	"""

	matrix_file_name = "DATA/data_formated.csv"
	label_file_name = "DATA/data_formated_label.csv"

	input_file_data = open(input_file_name, "r")
	matrix_file = open(matrix_file_name, "w")
	label_file = open(label_file_name, "w")

	cmpt = 0
	for line in input_file_data:
		lineWithoutBackN = line.split("\n")
		lineWithoutBackN = lineWithoutBackN[0]
		lineInArray = lineWithoutBackN.split(";")

		patient_identifiant = "undef"
		patient_diagnostique = "undef"
		patient_vector = ""

		if(cmpt > 0):
			index = 0
			for scalar in lineInArray:
				if(scalar != "0" and scalar != "1"):
					patient_identifiant = scalar

					index_file = open(index_file_name, "r")

					for index_file_line in index_file:
						index_file_lineWithoutBackN = index_file_line.split("\n")
						index_file_lineWithoutBackN = index_file_lineWithoutBackN[0]
						index_file_line_array = index_file_lineWithoutBackN.split(";")

						index_identifiant = index_file_line_array[0]
						index_diagnostique = index_file_line_array[1]

						if(index_identifiant == patient_identifiant):
							patient_diagnostique = index_diagnostique
					index_file.close()


				else:
					patient_vector += scalar + ";"
				index += 1

			# Write in output files
			patient_vector = patient_vector[:-1]
			matrix_file.write(patient_vector+"\n")
			label_file.write(patient_diagnostique+"\n")


		cmpt += 1

	label_file.close()
	matrix_file.close()
	input_file_data.close()