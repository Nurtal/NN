"""
main for NN project
"""

#-------------#
# IMPORTATION #######################################################################
#-------------#
import NeuralNetwork
import DataManager
import dichotomization
import sys
import shutil
import os
import numpy

#-------#
# TRASH ###########################################################################
#-------#

############################################
# Running neural network, the basic method #
############################################

command = sys.argv[1]






#-------------------#
#  => Prepare Data  #
#-------------------#

# -> Cytometry data
# - dichotomization of selected panels
# - save dihotomized matrix in a file
# - merge the files of interest
# - format the file to be process by the neural network
panels_to_process = ["1", "2", "3"]
number_of_intervals_for_dichtomization = 5

# => Dichotomization
for panel in panels_to_process:
	data_file_name = "../RD/sample/DATA/MATRIX/panel_"+str(panel)+"_filtered_processed.txt"
	
	# -> Generate matrix from data file
	description = dichotomization.extract_matrix_from(data_file_name)
	matrix = description[0]

	# -> create disjonct table for all variable in a matrix
	disjonct_tables = dichotomization.create_disjonctTable_for_matrix(matrix, number_of_intervals_for_dichtomization)

	# use disjonct table for dichotomization
	matrix_dichotomized = dichotomization.dichotomize(matrix, disjonct_tables)

	# save dichotomized matrix in a file to be processed by NN
	dichotomization.save_dichotomized_matrix_in_file(description[1], description[2], matrix_dichotomized, number_of_intervals_for_dichtomization, "../RD/sample/DATA/MATRIX/panel_"+str(panel)+"_dichotomized.csv")

# => Merge file
if(len(panels_to_process) > 1):
	print "[+] Merging files"
	input_file_name_1 = "../RD/sample/DATA/MATRIX/panel_"+str(panels_to_process[0])+"_dichotomized.csv"
	input_file_name_2 = "../RD/sample/DATA/MATRIX/panel_"+str(panels_to_process[1])+"_dichotomized.csv"
	DataManager.merge_input_files(input_file_name_1, input_file_name_2)

	if(len(panels_to_process) > 2):
		for panel_position in range(2, len(panels_to_process)):
			panel = panels_to_process[panel_position]
			shutil.copyfile("DATA/data_merged.csv", "DATA/data_merged.tmp")
			input_file_name_2 = "../RD/sample/DATA/MATRIX/panel_"+str(panel)+"_dichotomized.csv"
			DataManager.merge_input_files("DATA/data_merged.tmp", input_file_name_2)
			os.remove("DATA/data_merged.tmp")

			
# => Reformat file
DataManager.reformat_inputFile("DATA/data_merged.csv", "../RD/sample/DATA/patientIndex.csv")
PathToMatrixFile = "DATA/data_formated.csv"
PathToMatrixLabelFile = "DATA/data_formated_label.csv"


if(command == "main"):

	# Very basic way

	binaryClassification = 1
	sizeOfValidationSet = 60
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
	NeuralNetwork.TrainAndValidate(X, y, X_validation, y_validation, outputLayerSize, 1)



# TEST SPACE
if(command == "cross_validation"):
	binaryClassification = 1
	sizeOfValidationSet = 60
	numberOfPatient = DataManager.get_NumberOfPatients(PathToMatrixFile)
	numberOfSample = numberOfPatient / sizeOfValidationSet
	numberOfPatient = DataManager.get_NumberOfPatients(PathToMatrixFile)

	matrix_sets = DataManager.cross_validation(PathToMatrixFile, PathToMatrixLabelFile, sizeOfValidationSet, binaryClassification)

	X_sets = matrix_sets[0]
	X_validation_sets = matrix_sets[1]
	y_sets = matrix_sets[2]
	y_validation_sets = matrix_sets[3]


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
	score_list = []
	for x in range(0, numberOfSample):
		X = X_sets[x]
		X_validation = X_validation_sets[x]
		y = y_sets[x]
		y_validation = y_validation_sets[x]

		score = NeuralNetwork.TrainAndValidate(X, y, X_validation, y_validation, outputLayerSize, 0)
		score_list.append(score)

	# compute final score
	final_score = numpy.average(score_list)
	print final_score
