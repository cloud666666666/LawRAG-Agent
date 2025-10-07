from tools.models_load import llm_load
from transformers import GenerationConfig

llm, tokenizer = llm_load()
print(llm)
config = llm.config
new_config = GenerationConfig(
    max_new_tokens=100,
    do_sample=False,
    temperature=None,
    top_p=None,
    top_k=None,
    repetition_penalty=1.05,
    eos_token_id=getattr(tokenizer, "eos_token_id", None),
    pad_token_id=getattr(tokenizer, "pad_token_id", None) or getattr(tokenizer, "eos_token_id", None),
    use_cache=False
)
print(new_config)
device = next(llm.parameters()).device
inputs = tokenizer("Hello, world!", return_tensors="pt").to(device)
out = llm.generate(
    input_ids=inputs.input_ids,
    generation_config=new_config,
)
print(out)
