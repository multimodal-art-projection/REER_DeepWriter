import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import random
import os
from transformers import set_seed
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from typing import List
import pdb
import copy
import time
import numpy as np
import json
# Set your Hugging Face token here
os.environ["HUGGINGFACE_HUB_TOKEN"] = "hf_yourkey"

# For reproducibility
SEED = 1234
set_seed(SEED)
random.seed(42)

class LM:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-Math-7B-Instruct", model_type: str = "hf", model_url="", num_rollouts: int = 5, tokenizer=None, **model_args):
        self.model_type = model_type.lower()
        self.model_name = model_name
        
        self.max_tokens = model_args['max_tokens']
        self.temperature_range = model_args['temperature_range']
        self.topp = model_args['top_p']
        # self.topk = model_args['top_k']
        self.port = model_args.get("port", 0)
        self.do_bs = model_args.get('beamsearch', 0)
        self.url = model_url
        # if self.port!="0": 
        self.url = f"http://127.0.0.1:{self.port}/v1/completions"
        print(f"running server on {self.url}")
        self.num_rollouts = num_rollouts
        
        self.payload = {
            "model":self.model_name,
            "max_tokens": self.max_tokens,
            "top_p": self.topp,
            "n": self.num_rollouts
        }
        self.tokenizer = tokenizer
        
        self.__dict__.update(model_args)
        print("Updated model args:", self.__dict__)
        
        if self.model_type == "vllm":
            #raise NotImplementedError("VLLM is not implemented yet")
            from vllm import LLM, SamplingParams
            self.llm = LLM(model=model_name, enable_prefix_caching=True)
            self.SamplingParams = SamplingParams
        elif self.model_type == "hf":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name, torch_dtype=torch.float16, device_map="cuda"
            )
        elif self.model_type == "openai":
            import openai
            self.client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        elif self.model_type == "anthropic":
            import anthropic
            self.client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        elif self.model_type == "vllm_server":
            pass
        else:
            raise ValueError("Invalid model_type. Choose 'vllm', 'hf', 'openai', or 'anthropic'.")
        
    def generate(self, prompt, num_rollouts=None, isgreedy=False, **kwargs):
        if num_rollouts is None:
            num_rollouts = self.num_rollouts
        if self.model_type == 'vllm_server':
            return self.generate_vllm_server(prompt, num_rollouts, isgreedy=isgreedy, **kwargs)
        elif self.model_type == "vllm":
            return self.generate_vllm(prompt, num_rollouts)
        elif self.model_type == "hf":
            return self.generate_hf(prompt, num_rollouts)
        elif self.model_type == "anthropic" or self.model_type == "openai":
            return self.generate_api(prompt, num_rollouts)

    def generate_hf(self, prompt, num_rollouts):
        inputs = self.tokenizer(prompt, return_tensors="pt").to('cuda')
        print(prompt)
        results = []
        for _ in range(num_rollouts):
            temperature = random.uniform(self.temperature_range[0], self.temperature_range[1])
            outputs = self.model.generate(
                **inputs, do_sample=True, max_new_tokens=self.max_tokens, temperature=temperature,
                num_return_sequences=1
            )
            generated_tokens = outputs[0][inputs['input_ids'].shape[1]:]
            result = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            results.append(result)
        pdb.set_trace()
        return results

    def generate_vllm(self, prompt, num_rollouts):
        #raise NotImplementedError("VLLM is not implemented yet")
        # print(prompt)
        temperature = random.choice(self.temperature_range)
        sampling_params = self.SamplingParams(
            temperature=temperature,
            top_k=self.topk,
            top_p=self.topp, 
            max_tokens=self.max_tokens,
            n=num_rollouts,
            seed=SEED,
            # stop=['\n'],
        )
        st = time.time()
        outputs = self.llm.generate(prompt, sampling_params)
        ed = time.time()
        print(f'{num_rollouts} responses Time taken: {ed-st}')
        result = [completion.text for output in outputs for completion in output.outputs]
        return result, temperature
    
    def generate_vllm_server(self, prompt, num_rollouts=None, isgreedy=False, special_stop=None, prompt_only=False):
        
        temperature = np.random.uniform(low=self.temperature_range[0], high=self.temperature_range[1])
        # temperature = random.choice(self.temperature_range)
        payload = copy.copy(self.payload)
        
        payload.update({
            "temperature": temperature,
            # "messages": [
            #     {"role": "system", "content": systemprompt},
            #     {"role": "user", "content": query}
            # ],
            # "max_tokens": 4096,
            "prompt": prompt,
            'logprobs': 1 if not isgreedy else 0,
        })
        if num_rollouts is not None:
            payload['n'] = num_rollouts
            
        if isgreedy:
            payload['top_k'] = 1
            payload['top_p'] = 1
            payload['temperature'] = 0
            payload['n'] = 1
        
        if special_stop:
            payload['stop'] = special_stop
        
        if prompt_only:
            payload.update({
                'prompt': prompt, 
                'n': 1,
                'temperature': 1.0, 
                'prompt_logprobs': 1
            })
            
        # print(f'===> submitting request @{self.url}')
        response = requests.post(self.url,
                             headers={"User-Agent": "Test Client"},
                             json=payload,
                             stream=False)
        # {"object":"error","message":"[{'type': 'missing', 'loc': ('body', 'model'), 'msg': 'Field required'
        if response.status_code == 200:
            result = response.json()
            # print(num_rollouts, "Generated Text:", result)
        else:
            result = dict(choices=[])
            print(f"Error: {response.status_code}, {response.text}")
            return ([],[], []), None
            message = json.loads(response.text)['message']
            
            if 'context length' in message:
                encoded = self.tokenizer(prompt)
                q_tokens = len(encoded['input_ids'])
                a_tokens_max = 4096-5-q_tokens 
                payload['max_tokens'] = a_tokens_max
                
                response = requests.post(self.url,
                    headers={"User-Agent": "Test Client"},
                    json=payload,
                    stream=False
                )
                
                if response.status_code == 200:
                    result = response.json()
                    # print("Generated Text:", result)
                else:
                    result = dict(choices=[])
                    print(f"Error: {response.status_code}, {response.text}")
                
                
        result_ = [item['text'] for item in result['choices']]
        logps_ = [item['prompt_logprobs'] for item in result['choices']] if prompt_only else [item['logprobs']['token_logprobs'] for item in result['choices']]
        offsets_ = result if prompt_only else [item['logprobs']['text_offset'] for item in result['choices']]
        # offsets_ = [item['prompt_logprobs'] for item in result['choices']]
        
        offsets_ = result
        return (result_,logps_, offsets_), temperature

    def generate_vl_vllm_server(self, conversation, num_rollouts=None, isgreedy=False, special_stop=None, prompt_only=False):
        
        temperature = np.random.uniform(low=self.temperature_range[0], high=self.temperature_range[1])
        # temperature = random.choice(self.temperature_range)
        payload = copy.copy(self.payload)
        
        payload.update({
            "temperature": temperature,
            "messages": conversation,
            # "max_tokens": 4096,
            # "prompt": prompt,
            'logprobs': 1 if not isgreedy else 0,
        })

        if num_rollouts is not None:
            payload['n'] = num_rollouts
            
        if isgreedy:
            payload['top_k'] = 1
            payload['top_p'] = 1
            payload['temperature'] = 0
            payload['n'] = 1
        
        if special_stop:
            payload['stop'] = special_stop
        
            
        # print(f'===> submitting request @{self.url}')
        response = requests.post(self.url.replace("completions","chat/completions"),
                             headers={"User-Agent": "Test Client"},
                             json=payload,
                             stream=False)

        if response.status_code == 200:
            result = response.json()
            # print(num_rollouts, "Generated Text:", result)
        else:
            result = dict(choices=[])
            print(f"Error: {response.status_code}, {response.text}")
            # import pdb; pdb.set_trace()
            return ([],[], []), None
                
        result_ = [item['message']['content'] for item in result['choices']]
        logps_ = [[x['logprob'] for x in item['logprobs']['content']] for item in result['choices']]  
        offsets_ = None # result if prompt_only else [item['logprobs']['text_offset'] for item in result['choices']]
        # offsets_ = [item['prompt_logprobs'] for item in result['choices']]
        
        offsets_ = result
        return (result_,logps_, offsets_), temperature

    def generate_api(self, prompt: str, num_rollouts) -> List[str]:
        def send_request(prompt):
            temperature = random.choice(self.temperature_range)
            if self.model_type == "openai":
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=temperature
                )
                output = response.choices[0].message.content
            elif self.model_type == "anthropicc":
                response = self.client.messages.create(
                    model=self.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=self.max_tokens,
                    temperature=temperature
                )
                output = response.content[0].text
            return output

        responses = []
        with ThreadPoolExecutor(max_workers=num_rollouts) as executor:
            futures = [executor.submit(send_request, prompt) for _ in range(num_rollouts)]
            for future in tqdm(as_completed(futures), total=len(futures)):
                responses.append(future.result())

        return responses
