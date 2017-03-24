import DataManager
import NeuralNetwork
import sys

number_of_interval = sys.argv[1]
panel = sys.argv[2]

# => Reformat file
DataManager.reformat_inputFile("../RD/sample/DATA/MATRIX/data_dichotomized_pattern_"+str(number_of_interval)+".csv", "../RD/sample/DATA/patientIndex.csv")
PathToMatrixFile = "DATA/data_formated.csv"
PathToMatrixLabelFile = "DATA/data_formated_label.csv"

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
score = NeuralNetwork.TrainAndValidate(X, y, X_validation, y_validation, outputLayerSize)

# write result in log file
log_file = open("dichotomization_exploration_panel_"+str(panel)+".log", "a")
log_file.write(str(number_of_interval)+";"+str(float(score))+"\n")
log_file.close()