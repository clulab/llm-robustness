from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import argparse

import yaml

def negation_performance(labels_true, labels_pred, average=None):
    f1 = f1_score(y_true=labels_true, y_pred=labels_pred, average=average)
    precision = precision_score(y_true=labels_true, y_pred=labels_pred, average=average)
    recall = recall_score(y_true=labels_true, y_pred=labels_pred, average=average)

    return {"f1": f1, "precision": precision, "recall": recall}





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


if( dataInformation['dataset'] == "ner"):
    gold_binary =[]
    pred_binary =[]
    for item in data:
        gold_binary.append(1 if item['gold_answer'].lower().strip() == 'yes' else 0)
        pred_binary.append(1 if item['output'].lower().strip() == 'yes' else 0)

    neg_metrics = negation_performance(gold_binary, pred_binary)

    outputs = []
    accuracy = accuracy_score(gold_binary, pred_binary)
    outputs.append({"permutation": dataInformation['permutation'], "model": dataInformation['model'], "precision": neg_metrics['precision'], "dataset": dataInformation['input'], "negation_metrics": neg_metrics})

else:
    y_true =[]
    y_pred =[]
    for item in data:
        gold_answer = item['gold_answer'].lower()
        y_true.append(gold_answer)
        y_pred.append(item['output'].lower())  

    outputs =[]
    accuracy = accuracy_score(y_true, y_pred)
    print(f"Accuracy: {accuracy:.3f}")

    if( dataInformation['dataset'] != "ner"):
        outputs.append({"permutation": dataInformation['permutation'], "model": dataInformation['model'], "accuracy": accuracy, "dataset": dataInformation['input']})





with open("eval_output.json", "a", encoding="utf-8") as f:
    json.dump(outputs, f)
    f.write("\n")  


