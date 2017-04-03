"""
Script to parse the downloaded hotel reviews data and produce an output file.
"""
import os 
import re 

## Config Params 
_dataset_path = 'data/hotel_data/'
_output_path = 'data/hotel_reviews/'
_num_hotels = 2

## Review Files
files = os.listdir(_dataset_path)
for raw_file in files[:_num_hotels]:

	## Output path 
	reivew_path = _output_path + raw_file.replace(".dat","") +"_"+ "reviews.txt"
	output_file = open(reivew_path, "w")

	## Parse Review Text 
	content = open(_dataset_path+raw_file).read()
	indexes =  re.finditer('<Content>', content)
	review_lines = []

	for index in indexes:
		_start_ind = index.start()
		for line in content[_start_ind:].split("\n"):
			if line.startswith("<Date>"):
				break
			if not "showReview" in line and line.strip():
				review_lines.append(line)

	## Generate Output Files
	review_text = "\n".join(review_lines)
	review_text = review_text.replace("<Content>", "")
	output_file.write(review_text + "\n")