import ast 

companies = open("../data/meta_data/company_list.txt").read().strip().split("\n")
companies = [company.lower().replace(" ","") for company in companies]
data = open("../data/twitter_data.txt").read().strip().split("\n")

visited = []
for doc in data:
	try:
		doc = ast.literal_eval(doc)
		name = doc['name'].lower().replace(" ","")
		desc = doc['description']

		for fil in companies:
			if fil == name and desc and name not in visited:
				visited.append(name)
				print name + "||" + " ".join(desc.split())
				break

	except Exception as E:
		print E 
		pass 