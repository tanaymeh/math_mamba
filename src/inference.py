import random
import torch
from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = ""

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map='auto',
    torch_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)

pipe = pipeline('text-generation', model=model, tokenizer=tokenizer)

val_dataset = load_dataset('json', data_files='data/valid_dataset.jsonl', split='train')
idx = random.randint(0, len(val_dataset))
sample = val_dataset[idx]["messages"][:2]
prompt = pipe.tokenizer.apply_chat_template(sample, tokenize=False, add_generation_prompt=True)

outputs = pipe(
    prompt, 
    max_new_tokens=256, 
    do_sample=True, 
    temperature=0.1, 
    top_k=50, 
    top_p=0.1, 
    eos_token_id=pipe.tokenizer.eos_token_id,
    pad_token_id=pipe.tokenizer.pad_token_id
)

print(f"Query:\n{val_dataset[idx]['messages'][1]['content']}")
print(f"Original Answer:\n{val_dataset[idx]['messages'][2]['content']}")
print(f"Generated Answer:\n{outputs[0]['generated_text'][len(prompt):].strip()}")

# Push model to hub
from huggingface_hub import notebook_login
notebook_login()

model.push_to_hub("")