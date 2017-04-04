"""
Score function for the AG exploration
-> use support vector machine as a score
function for the exploration discretization
with genetic algorithm

"""

from sklearn import tree
import platform
import DataManager
import numpy
import sys



# => filter data
strategy_filter = sys.argv[1]
DataManager.filter_input_data("../RD/sample/DATA/MATRIX/data_dichotomized_pattern_individual_to_evaluate.csv", "../RD/sample/DATA/patientIndex.csv", strategy_filter)

# => Reformat file
if(platform.system() == "Windows"):
	DataManager.reformat_inputFile("..\\RD\\sample\\DATA\\MATRIX\\data_dichotomized_pattern_individual_to_evaluate.csv", "..\\RD\\sample\\DATA\\patientIndex.csv")
elif(platform.system() == "Linux"):
	DataManager.reformat_inputFile("../RD/sample/DATA/MATRIX/data_dichotomized_pattern_individual_to_evaluate_filtered.csv", "../RD/sample/DATA/patientIndex.csv")

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


score_list = []
for x in range(0, numberOfSample):
	X = X_sets[x]
	X_validation = X_validation_sets[x]
	y = y_sets[x]
	y_validation = y_validation_sets[x]


	# classification & prediction
	clf = tree.DecisionTreeClassifier()
	clf = clf.fit(X, y)

	prediction = clf.predict(X_validation)

	# evaluate score
	cmpt = 0
	good_answer = 0
	for diag_predicted in prediction:
		diag_real = y_validation[cmpt]

		if(diag_real == diag_predicted):
			good_answer += 1

		cmpt+=1


	score = float(good_answer) / float(len(y_validation))
	score = score*100
	score_list.append(score)


# compute final score
final_score = numpy.average(score_list)

# write result in log file
log_file = open("evaluation_score.log", "w")
log_file.write(str(float(score))+"\n")
log_file.close()