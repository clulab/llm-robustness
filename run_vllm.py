import json 
import yaml
import os 
from vllm import LLM, SamplingParams

#Qwen/Qwen3-0.6B local model to run on compurter
#Qwen/Qwen2.5-VL-3B-Instruct

def main():
        # Open the file
        with open('sentiment_analysis.json', 'r+') as file:
         #load the file as a dic
                data = json.load(file)

       
        #get rid of this below
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

        config = yaml.safe_load(open('qwen_ptfeq.yaml'))

    

        with open(config['input']) as f:
                data = [json.loads(line) for line in f if line.strip()]

        print(json.dumps(data, indent=4))
   

        combined_string = ''.join([DATA[ch] for ch in config['permutation']])

        print(config['permutation'])

        model = LLM(
                model = config['model'], seed =1 , max_model_len=4096
                )
        sampling_params = SamplingParams(n=1, temperature=0, max_tokens=100, stop=["\n"])

     #for loop in here 
        output =model.generate(combined_string , sampling_params =sampling_params)

        outputs =output[0].outputs[0].text

        

        total_outputs = combined_string + outputs + '\n' + config['permutation'] + '\n' + config['model']
        lines = total_outputs.splitlines()
        print("Into the json we go ")
        with open("output.json","w", encoding= "utf-8") as json_file:
                print("Dumping into json")
                json.dump(lines, json_file, indent= 4)
             


#so the model dosen't crash on me               
if __name__ == "__main__":
        main()

#to-do 
#Work on the combine string to make sure it can replace the input part as well
#In the model.generate part make sure that the input is getting changed with .replace and whatever the input is 
#Work on the for loop to make sure it goes around the generate part
# add the path to all the config files and change it in the code so it reads up on to that 
#Then work on the output make it be the input and add it with input should be id input gold_answer and output 
