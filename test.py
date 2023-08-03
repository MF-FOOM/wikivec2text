import os
from contextlib import nullcontext
import torch
import openai
import tiktoken
from model import GPTConfig, GPT
from huggingface_hub import hf_hub_download

enc = tiktoken.get_encoding("gpt2")
openai.api_key = os.getenv("OPENAI_API_KEY")

max_new_tokens = 100
temperature = 0.1
device = 'cuda'
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16'
exec(open('configurator.py').read())  # overrides from command line or config file

seed = 1337
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

checkpoint = torch.load(hf_hub_download(repo_id="MF-FOOM/wikivec2text", filename="ckpt.pt"), map_location=device)
model = GPT(GPTConfig(**checkpoint['model_args']))
state_dict = checkpoint['model']
unwanted_prefix = '_orig_mod.'
for k,v in list(state_dict.items()):
    if k.startswith(unwanted_prefix):
        state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)

model.load_state_dict(state_dict)

model.eval()
model.to(device)

def embed(string):
    return torch.tensor(openai.Embedding.create(
        model="text-embedding-ada-002",
        input=string
    )['data'][0]['embedding'], device=device)

with torch.no_grad():
    starting_seq = "Revolver is the seventh album by the English rock band the Beatles."
    seq_to_sub = "The Beatles are a rock band."
    seq_to_add = "The Beatles are a classical band."

    embedding = embed(starting_seq) - embed(seq_to_sub) + embed(seq_to_add)

    print(enc.decode(model.generate(embedding, max_new_tokens, temperature=temperature).tolist()))
