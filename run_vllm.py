import argparse
from email import parser
import json 
import yaml
from vllm import LLM, SamplingParams

#this is for the local machine, if you want to run it on the server, leave as is
#import vllm.envs as envs
#envs.VLLM_HOST_IP="0.0.0.0" or "127.0.0.1"

#Qwen/Qwen3-0.6B local model to run on compurter
#Qwen/Qwen2.5-VL-3B-Instruct


def main():
        # Open the file
        with open(config['prompt'], 'r+') as file:
         #load the file as a dic
                data = json.load(file)

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

        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--config', default='qwen_ptfeq.yaml')
        args = parser.parse_args()
        with open(args.config) as f:
          config = yaml.safe_load(f)





        print(config['permutation'])

        with open(config['input']) as f:
                data = [json.loads(line) for line in f if line.strip()]

        #Making saure it can read the json and puts it in the terminal
        #print(json.dumps(data, indent=4))
   

        combined_string = ''.join([DATA[ch] for ch in config['permutation']])

        #Permutation check
        #print(config['permutation'])

        model = LLM(
                model = config['model'], seed =1 , max_model_len=2048, gpu_memory_utilization=0.8
                )
        sampling_params = SamplingParams(n=1, temperature=0, max_tokens=4)


        for item in data:
                if item['input'] == "/media/volume/llm-robustness-data/datasets/sentiment-analysis/drugCom/drugCom_toy.jsonl":
                        if item['id'] == 80454:
                                continue
                #print(item['input'])
                #print(combined_string.replace('<input>', item['input']))
                output =model.generate(combined_string.replace('<input>', item['input']) , sampling_params =sampling_params)

                outputs =output[0].outputs[0].text

                new_outputs = {"id": item['id'], "input": item['input'], "gold_answer": item['gold_answer'], "output": outputs}
                
                with open("output.json", "a", encoding="utf-8") as f:
                        json.dump(new_outputs, f)
                        f.write("\n")
             


#so the model dosen't crash on me               
if __name__ == "__main__":
        main()


