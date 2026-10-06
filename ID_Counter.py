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

path = "/media/volume/llm-robustness-data/datasets/sentiment-analysis/drugCom/drugCom_toy.jsonl"

counts = Counter()
with open(path) as f:
    for line in f:
        line = line.strip()
        if not line:          # skip blank lines
            continue
        item = json.loads(line)
        counts.update(collect_values(item.get("gold_answer", [])))

print(counts)