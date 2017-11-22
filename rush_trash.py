

data_file = open("C:\\Users\\NaturalKiller01\\Desktop\\Nathan\\Spellcraft\\FlowCytometry_data_NA_filtered.csv", "r")
output_file = open("cyto_data_old.csv", "w")
for line in data_file:
	
	line = line.replace("\n", "")
	line = line.replace("\"", "")
	line_in_array = line.split(",")
	
	keep_the_line = True
	for scalar in line_in_array:
		if(scalar == ""):
			keep_the_line = False
	if(keep_the_line):
		output_file.write(line+"\n")

output_file.close()
data_file.close()