import json
from collections import Counter

def collect_values(obj):
    
    if isinstance(obj, dict):
        for v in obj.values():
            yield from collect_values(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from collect_values(v)
    else:
        yield obj

with open("/media/volume/llm-robustness-data/datasets/sentiment-analysis/drugCom/drugCom_toy.jsonl") as f:
    data = [json.loads(line) for line in f]

counts = Counter(collect_values(data["gold_answers"]))
print(counts)