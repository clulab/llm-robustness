import re
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import json
import argparse

import yaml



def postprocess_entities(text: str ) -> set:
    match = re.search(r'<output>(.*?)</output>', text, re.DOTALL)
    if match: 
        content = match.group(1).strip()
        entities = {item.strip() for item in content.split('\n') if item.strip()}
        return entities


    return set()



def negation_performance(labels_true, labels_pred, average=None):
 tp = 0 
 fp = 0
 fn = 0
 predictions =[]

 for item in data:
    gold = postprocess_entities(item['gold_answer'])
    predicted = postprocess_entities(item['output'])

    sample_tp = len(gold & predicted)
    sample_fp = len(predicted - gold)
    sample_fn = len(gold - predicted)

    sample_precision = sample_tp / (sample_tp + sample_fp) if (sample_tp + sample_fp) > 0 else 0
    sample_recall = sample_tp / (sample_tp + sample_fn) if (sample_tp + sample_fn) > 0 else 0
    sample_f1 = 2 * sample_precision * sample_recall / (sample_precision + sample_recall) if (sample_precision + sample_recall) > 0 else 0



    tp += sample_tp
    fp += sample_fp
    fn += sample_fn

    prediction_entry = {
        'question': item['question'],
        'gold_answer': item['gold_answer'],
        'predicted_answer': item['output'],
        'processed_gold_answer': list(gold),
        'processed_predicted_answer': list(predicted),
        'f1_score': sample_f1
    }
    predictions.append(prediction_entry)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1_score': round(f1, 4),
        'predictions': predictions #dont know if needed
    }


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
    neg_metrics = negation_performance(data)

    outputs = []
    accuracy = neg_metrics['f1_score']
    outputs.append({"permutation": dataInformation['permutation'], "model": dataInformation['model'], "precision": neg_metrics['precision'], "dataset": dataInformation['input'], "negation_metrics": neg_metrics, "recall": neg_metrics['recall'], "f1_score": neg_metrics['f1_score']})

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


