"""
main for NN project
"""

#-------------#
# IMPORTATION #######################################################################
#-------------#
import NeuralNetwork
import DataManager
import sys


#-------#
# TRASH ###########################################################################
#-------#

############################################
# Running neural network, the basic method #
############################################

# Prepare Data
DataManager.merge_input_files("../RD/sample/DATA/MATRIX/panel_1_dichotomized.csv", "../RD/sample/DATA/MATRIX/panel_2_dichotomized.csv")
DataManager.reformat_inputFile("DATA/data_merged.csv", "../RD/sample/DATA/patientIndex.csv")
PathToMatrixFile = "DATA/data_formated.csv"
PathToMatrixLabelFile = "DATA/data_formated_label.csv"

binaryClassification = 0
sizeOfValidationSet = 30
numberOfPatient = DataManager.get_NumberOfPatients(PathToMatrixFile)
matrix = DataManager.generate_TrainingAndValidationMatrix(PathToMatrixFile, sizeOfValidationSet)
labels = DataManager.generate_LabelsVectors(PathToMatrixLabelFile, binaryClassification, sizeOfValidationSet, numberOfPatient)
X = matrix[0]
X_validation = matrix[1]
y = labels[0]
y_validation = labels[1]


# Define few parameter (only one actually)
# for the neural network
outputLayerSize = "undef"
if(binaryClassification):
	outputLayerSize = 2
else:
	distinctValue = []
	for element in y:
		if element not in distinctValue:
			distinctValue.append(element)
	outputLayerSize = int(len(distinctValue))

# Run the Neural Network
NeuralNetwork.TrainAndValidate(X, y, X_validation, y_validation, outputLayerSize)