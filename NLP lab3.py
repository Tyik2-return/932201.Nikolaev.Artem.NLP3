from transformers import BertTokenizer, BertForMaskedLM
from torch.nn import functional as F
import torch

name = 'bert-base-multilingual-uncased'
tokenizer = BertTokenizer.from_pretrained(name)
model = BertForMaskedLM.from_pretrained(name, return_dict=True)

text = "Магазин находится " + tokenizer.mask_token + " большого торгового центра."
input = tokenizer.encode_plus(text, return_tensors="pt")
mask_index = torch.where(input["input_ids"][0] == tokenizer.mask_token_id)[0]

with torch.no_grad():
    output = model(**input)
logits = output.logits
softmax = F.softmax(logits, dim=-1)
mask_word_probs = softmax[0, mask_index, :]
top_10 = torch.topk(mask_word_probs, 10)

print("Топ-10 слов:")
for i, token_id in enumerate(top_10.indices[0]):
    token = tokenizer.decode([token_id]).strip()
    print(f"{i+1:2}. {token}")
