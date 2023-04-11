#
# file = open("/content/drive/MyDrive/FLLM/train_sentences/all_sentences.txt", 'r')
# file_data = file.readlines()
# file.close()
# len(file_data)
import torch
from spacy.lang.en import English
import random
from transformers import GPT2LMHeadModel, GPT2TokenizerFast,GPT2Tokenizer
from sklearn.model_selection import train_test_split
import numpy as np
from pprint import pprint
from tqdm import tqdm
import math
import zlib
zlib_sentences=[]
file = open("/data/perplexity_ranking/PPL-XL_Zlib.txt", 'r')
result1=file.readlines()
for i in range(1000):
  zlib_sentences.append(result1[i*3+2])

TRAIN_DATA_FOLDER_PATH="../../FLLM_black_box2/data/train_sentences/"
train_data=[]
for i in range(3):
    file_name = f"client_{i + 1}.txt"
    file_path = TRAIN_DATA_FOLDER_PATH + file_name
    file = open(file_path, 'r')
    a = file.readlines()
    train_data.extend(a)
len(train_data)

nlp = English()
spacy_tokenizer = nlp.tokenizer
tokens = spacy_tokenizer(zlib_sentences[12])

def get_k_tokens_set(sequence,k):
    k_tokens_list=[]
    n=len(sequence)
    for i in range(n-2):
      a=sequence[i:i+k]
      k_tokens_list.append(a)
    # print(len(k_tokens_list))
    return k_tokens_list

def get_match_score(tokens_x, tokens_y):
    score=0
    common_list=[]
    for i in range(len(tokens_x)):
        a = tokens_x[i]
        for j in range(len(tokens_y)):
            b = tokens_y[j]
            if a == b:
              if a not in common_list:
                score+=1
                common_list.append(a)
    return score,common_list

k=3
m=len(zlib_sentences)
n=len(train_data)

scores=np.zeros((m,n), dtype = float)
print(scores.shape)
count=0
for i in range(m):
  
  count+=1
  if count%10==0:
    print(count)
  x=zlib_sentences[i]
  tokens_x=get_k_tokens_set(spacy_tokenizer(x),k)
  for j in range(n):
    y=train_data[j]
    tokens_y=get_k_tokens_set(spacy_tokenizer(y),k)
    score,_=get_match_score(tokens_x,tokens_y)
    scores[i][j]=score

a=np.ravel(scores)
# print(a[:200])
ranked_indices = np.argsort(a)
reverse_ranked_indices = ranked_indices[::-1]

# largest_indices=largest_indices[:200]
a[ranked_indices]
meaningful_indices=[]
for i in reverse_ranked_indices:
  if a[ranked_indices]>0:
      meaningful_indices.add(i)

# for i in largest_indices:
#   j,k=np.unravel_index(i, scores.shape)
#   print(zlib_sentences[j],"\n",train_data[k])
#   print(scores[j][k])
client_name="client_1_3"
meaningful_indices=[]
# with open('test1.txt', 'w') as f:
#   for index in largest_indices:
#     f.write(f"{scores[j][k]}\n")
#     f.write(f"{zlib_sentences[j]}\n")
#     f.write(f"{train_data[j]}\n")
#     f.write("\n")
# f.close()

