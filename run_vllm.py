import json
import copy
import torch 
import os
import yaml
from vllm import LLM, SamplingParams

#Qwen/Qwen3-0.6B local model to run on compurter
#Qwen/Qwen2.5-VL-3B-Instruct

# Open the file
with open('sentiment_analysis.json', 'r+') as file:
    
   #load the file as a dic
    data = json.load(file)

# We change the input of the file
data['query'] = data['query'].replace('<input>', 'These pills are having no effect on me at all')

# Grabs all the information from the dic and turns it all into strings
persona_text = data.get("persona")
task_text = data.get("task")
format_text =data.get("format")
example_text = data.get("examples")
query_text = data.get("query")


example_text = '\n'.join(example_text)
DATA = {
        "p": persona_text,
        "t": task_text,
        "f": format_text,
        "e": example_text,
        "q": query_text

}

config = yaml.safe_load(open('Config.yaml'))


#permutation_py = input(config['permutation']).strip().lower()

#combined_string = ''.join([DATA[ch] for ch in permutation_py])



model = LLM(
        model = config['model'], seed =1 , max_model_len=37440
        )
sampling_params = SamplingParams(n=1, temperature=0, max_tokens=100, stop=["\n"])

#for i(len of the input json) and have the ai model re running it non stop but having it 

#load the data here mess with the input and then do a for a loop so it keep going thru the generate part then also have it output it as a json
output =model.generate(config['permutation'] , sampling_params =sampling_params)

outputs =output[0].outputs[0].text

#print(type((outputs)))

total_outputs = combined_string + outputs
total_outputs = json.dumps(total_outputs)
print('\n')
print(total_outputs)
with open("output.json","w", encoding= "utf-8") as json_file:
        json_file.write(total_outputs)
        #json.dump(combined_string, json_file, indent= 4, ensure_ascii=False)
        #json.dump(outputs,json_file)
