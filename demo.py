import DataManager
import NeuralNetwork
import sys
import numpy

number_of_interval = sys.argv[1]
panel = sys.argv[2]

# => Reformat file
DataManager.reformat_inputFile("../RD/sample/DATA/MATRIX/data_dichotomized_pattern_"+str(number_of_interval)+".csv", "../RD/sample/DATA/patientIndex.csv")
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

# write result in log file
log_file = open("dichotomization_exploration_panel_"+str(panel)+".log", "a")
log_file.write(str(number_of_interval)+";"+str(float(score))+"\n")
log_file.close()