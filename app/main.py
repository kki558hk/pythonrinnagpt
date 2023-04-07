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
prompt = "早く日本に帰りたいな。"
# 生成する文章の数
num = 1

input_ids = tokenizer.encode(
    prompt, return_tensors="pt", add_special_tokens=False).to(type)
with torch.no_grad():
    output = model.generate(
        input_ids,
        max_length=300,  # 最長の文章長
        min_length=50,  # 最短の文章長
        do_sample=True,
        top_k=600,  # 上位{top_k}個の文章を保持
        top_p=0.90,  # 上位{top_p}%の単語から選択する。例）上位95%の単語から選んでくる
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        num_return_sequences=num  # 生成する文章の数
    )
decoded = tokenizer.batch_decode(output, skip_special_tokens=True)
for i in range(num):
    print(decoded[i])
