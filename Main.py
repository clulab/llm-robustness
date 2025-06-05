import json
import copy

# Open file
file = open('sentiment_analysis.json')

# returns JSON object as a dictionary
data = json.load(file)

#in case we want to make a copy of the json and change that copy
# data_copy = copy.deepcopy(data)
#data_copy['query'] = data_copy['query'].replace('<input>', 'Hello')

# replace what the input is we want in the query
data['query'] = data['query'].replace('<input>', 'Hello')

#copy printing
#print(data_copy['query'])

#debugging
#print('\n')

# print the query data
print(data['query'])

# close json file
file.close()

