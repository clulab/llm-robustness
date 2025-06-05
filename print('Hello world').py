import json


f = open('sentiment_analysis.json')

# returns JSON object as a dictionary
data = json.load(f)

# Iterating through the json list
print(data['persona'])
print('\n')
print(data['task'])
print('\n')
print(data['format'])
print('\n')
print(data['examples'])
print('\n')
print(data['query'])
print('\n')

# Closing file
f.close()