import torch
from transformers import T5Tokenizer, AutoModelForCausalLM
from transformers import GPT2Tokenizer

tokenizer = T5Tokenizer.from_pretrained("rinna/japanese-gpt2-small")

model = AutoModelForCausalLM.from_pretrained("rinna/japanese-gpt2-small")


if torch.cuda.is_available():
    model = model.to("cuda")

type = 'cpu'
if torch.cuda.is_available():
    type = 'cuda'

model = model.to(type)

# 初めの文章
prompt = "私香港に"
# 生成する文章の数
num = 1

input_ids = tokenizer.encode(
    prompt, return_tensors="pt", add_special_tokens=False).to(type)
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_length=100,
        min_length=100,
        do_sample=True,
        top_k=500,
        top_p=0.95,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_return_sequences=num  # 生成する文章の数
    )
decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
for i in range(num):
    print(decoded[i])
