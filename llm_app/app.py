import torch 
from transformers import LlamaTokenizer, LlamaForCausalLM
from ray import serve 
from starlette.requests import Request 

model_path = 'openlm-research/open_llama_3b_v2'

@serve.deployment 
class TextGenerationModel:
    def __init__(self, model_path):
        self.tokenizer = LlamaTokenizer.from_pretrained(model_path)
        self.model = LlamaForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16, device_map="cpu")

    def generate(self, prompt: str, max_new_tokens: int = 32):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids 
        start_index = input_ids.shape[-1]
        generation_output = self.model.generate(input_ids=input_ids, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(generation_output[0][start_index:], skip_special_tokens=True)
    
    async def __call__(self, http_request: Request) -> str:
        prompt: str = await http_request.json()
        return self.generate(prompt)
    
app = TextGenerationModel.bind(model_path)