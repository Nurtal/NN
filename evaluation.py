import DataManager
import NeuralNetwork
import sys
import numpy
import platform



# => filter data
strategy_filter = sys.argv[1]
DataManager.filter_input_data("../RD/sample/DATA/MATRIX/data_dichotomized_pattern_individual_to_evaluate.csv", "../RD/sample/DATA/patientIndex.csv", strategy_filter)

# => Reformat file
if(platform.system() == "Windows"):
	DataManager.reformat_inputFile("..\\RD\\sample\\DATA\\MATRIX\\data_dichotomized_pattern_individual_to_evaluate.csv", "..\\RD\\sample\\DATA\\patientIndex.csv")
elif(platform.system() == "Linux"):
	DataManager.reformat_inputFile("../RD/sample/DATA/MATRIX/data_dichotomized_pattern_individual_to_evaluate.csv", "../RD/sample/DATA/patientIndex.csv")

PathToMatrixFile = "DATA/data_formated.csv"
PathToMatrixLabelFile = "DATA/data_formated_label.csv"

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


# Run the Neural Network
score_list = []
for x in range(0, numberOfSample):
	X = X_sets[x]
	X_validation = X_validation_sets[x]
	y = y_sets[x]
	y_validation = y_validation_sets[x]



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


	score = NeuralNetwork.TrainAndValidate(X, y, X_validation, y_validation, outputLayerSize, 0)
	score_list.append(score)

# compute final score
final_score = numpy.average(score_list)

# write result in log file
log_file = open("evaluation_score.log", "w")
log_file.write(str(float(score))+"\n")
log_file.close()