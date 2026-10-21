import json
import random 

dataset = ""

with open (dataset, "r") as f:
    data = json.load(f)



sample_size = 1000
random_sample = random.sample(data, sample_size)

file_name = "test.json"
with open(file_name, "w") as f:
    json.dump(random_sample, f, indent=4)
