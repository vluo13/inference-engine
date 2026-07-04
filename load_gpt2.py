from transformers import GPT2LMHeadModel
from transformers import pipeline

model_hf = GPT2LMHeadModel.from_pretrained("gpt2")
sd_hf = model_hf.state_dict()
for k, v in sd_hf.items():
    print(k, v.shape)

generator = pipeline("text-generation", model = "gpt2")
generator("Hello", max_length=10)
