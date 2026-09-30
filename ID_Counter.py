import json

count = 0
with open("yourfile.jsonl") as f:
    for line in f:
        if line.strip():
            json.loads(line)  # parse each line separately
            count += 1
print(count)