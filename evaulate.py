from sklearn.metrics import accuracy_score
import json
import argparse

import yaml


parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config', default='qwen_ptfeq.yaml')
args = parser.parse_args()
with open(args.config) as f:
    dataInformation = yaml.safe_load(f)



data = []
with open("output.json", "r") as file:
    for line in file:
        line = line.strip()
        if line:
            data.append(json.loads(line))

y_true =[]
y_pred =[]
for item in data:
    y_true.append(item['gold_answer'])
    y_pred.append(item['output'].split()[0])  # Assuming the model's output is a string and we want the first word as the prediction


outputs =[]
accuracy = accuracy_score(y_true, y_pred)
print(f"Accuracy: {accuracy:.3f}")
outputs.append({"permutation": dataInformation['permutation'], "model": dataInformation['model'], "accuracy": accuracy, "dataset": dataInformation['input']})



with open("eval_output.json", "a", encoding="utf-8") as f:
    json.dump(outputs, f)
    f.write("\n")  


'''
## to do
Need to get jet stream up and running again 

CSV 
permutation model accuarcy dataset
^^all the different titles





we can remove p t and never remove the e and the f

we can compare t vs e 

we remove pieces from the best accuracy one
'''