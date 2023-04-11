

import torch
import random
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from sklearn.model_selection import train_test_split

import numpy as np
from pprint import pprint
from tqdm import tqdm
import torch
import pandas as pd

import nltk
nltk.download('punkt')

#@title load data from email dataset
device = torch.device("cpu")
model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2", bos_token="<|startoftext|>", eos_token="<|endoftext|>",
                                              pad_token="<|pad|>")
model.resize_token_embeddings(len(tokenizer))

MODEL_PATH= "/model/gpt2/fed_avg_batch_30_clients_100_percent/server_2512.pt"
model.load_state_dict(
    torch.load(MODEL_PATH, map_location=device))

seq_len=50
model.eval()
def generate2(batch_size,num_samples,prompt,top_k=50,temperature=0.5):

  samples=[]
  num_batches = int(np.ceil(num_samples/batch_size))

  with tqdm(total=num_samples) as pbar:
    for i in range(num_batches):
      prompts = [prompt] * batch_size
      input_len = 1
      inputs = tokenizer(prompts, return_tensors="pt", padding=True)
      output_sequences = model.generate(
                input_ids=inputs['input_ids'].to(device),
                attention_mask=inputs['attention_mask'].to(device),
                max_length=input_len + seq_len,
                do_sample=True, 
                top_p=0.8,
                top_k=top_k,
                temperature=temperature
            )
      texts = tokenizer.batch_decode(output_sequences, skip_special_tokens=True)
      for text in texts:
        samples.append(text)
      pbar.update(batch_size)
  return samples



prompt="If you have any further questions, please give me a call at"
# file1 = open("/content/drive/MyDrive/FLLM/generation/50_0.8_tem_0.5_2000.txt", 'w')
# texts=generate2(50,2000,prompt,50,0.7)
inputs = tokenizer(prompt, return_tensors="pt", padding=True)
print(inputs)
def top_k_logits(logits, k):
    """
    Masks everything but the k top entries in the given logits.
    """
    if k == 0:
        # If k is 0, return logits as is.
        return logits
    else:
        # Get the top k largest values in the logits along the last dimension.
        values, _ = torch.topk(logits, k, dim=-1)

        # Create a mask that is 1.0 for the top k values and 0.0 for all others.
        mask = torch.gt(logits, values[..., -1:])

        # Set all values outside the top k to -infinity (or an appropriate minimum value to avoid overflow).
        masked_logits = logits.masked_fill(~mask, -float('inf'))

        return masked_logits

initial_temperature = 1
decay_rate = 0.9
min_temperature = 0.5

sentences=[]


def generate_one_by_one():
  prompt="If you have any further questions, please give me a call at"
  next_token_id=256
  length=0
  while next_token_id != tokenizer.eos_token_id and length<40 :
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    temperature = max(initial_temperature * decay_rate ** length, min_temperature)
    # temperature=0.5
    # print(temperature)

    outputs = model(input_ids=input_ids)
    next_token_logits = outputs[0][:, -1, :] / temperature
    filtered_logits = top_k_logits(next_token_logits, k=50)
    probs = torch.softmax(filtered_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    next_token_id=next_token.item()
    generated_text = tokenizer.decode(next_token_id)
    # print(generated_text)
    prompt+=generated_text
    length+=1
  return prompt

prompt="If you have any further questions, please give me a call at"

# inputs = tokenizer(prompt, return_tensors="pt", padding=True)
# print(inputs["input_ids"])
# avb=[]
# for i in range(1000):
#     if i%50==0:
#         print(i)
#     sen=generate_one_by_one()
#     avb.append(sen)
# generation = "/Users/macm2/PycharmProjects/FLLM/data/decaying.txt"
# file2 = open(generation, 'w')
# for text in avb:
#   text=text+"\n"
#   file2.write(text)
# file2.close()

