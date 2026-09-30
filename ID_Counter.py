import json

with open("/media/volume/llm-robustness-data/datasets/negation/i2b2/toy.jsonl") as f:
    data = json.load(f)
print(len(data))