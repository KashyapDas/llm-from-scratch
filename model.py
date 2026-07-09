import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import os
import time

# Assign the GPU if avalaible
device = "cuda" if torch.cuda.is_available() else "cpu"

# Loading the dataset
dataSet = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.exists("dataset.txt"):
    import urllib.request
    urllib.request.urlretrieve(dataSet,"dataSet.txt")
with open("dataSet.txt","r") as f:
    text = f.read()

# Tokenization
char = sorted(set(text))
vocab_len = len(char)
# Character to Integer Mapping
char_to_idx = {}
for i,c in enumerate(char):
    char_to_idx[c] = i
# Integer to Character Mapping
idx_to_char = {}
for i,c in enumerate(char):
    idx_to_char[i] = c
# Encode/Decode Function
def encode(string):
    result = []
    for c in string:
        result.append(char_to_idx[c])
    return result
def decode(arr):
    result = ""
    for i in arr:
        result = result + idx_to_char[i]
    return result

