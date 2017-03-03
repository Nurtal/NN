"""
main for NN project
"""

#-------------#
# IMPORTATION #######################################################################
#-------------#
from NeuralNetwork import *
from DataManager import *
import sys


#-------#
# TRASH ###########################################################################
#-------#

############################################
# Running neural network, the basic method #
############################################

# Prepare Data
#PathToMatrixFile = "../RD/sample/DATA/MATRIX/discrete_processed_binary.csv"
#PathToMatrixLabelFile = "../RD/sample/DATA/MATRIX/discrete_processed_binary_label.csv"
PathToMatrixFile = "DATA/data_formated.csv"
PathToMatrixLabelFile = "DATA/data_formated_label.csv"

binaryClassification = 0
sizeOfValidationSet = 30
numberOfPatient = get_NumberOfPatients(PathToMatrixFile)
matrix = generate_TrainingAndValidationMatrix(PathToMatrixFile, sizeOfValidationSet)
labels = generate_LabelsVectors(PathToMatrixLabelFile, binaryClassification, sizeOfValidationSet, numberOfPatient)
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
TrainAndValidate(X, y, X_validation, y_validation, outputLayerSize)