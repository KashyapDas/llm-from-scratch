import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import math
import os
import time
from rotary_embedding_torch import RotaryEmbedding   # Use for ROPE (Positional Encoding)
from xformers.ops import SwiGLU # Use for Feed Forwarding Layer

# Assign the GPU if avalaible
device = "cuda" if torch.cuda.is_available() else "cpu"

# Loading the dataset
dataSet = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
if not os.path.exists("dataSet.txt"):
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


# Spliting the data - Training/Validating Data
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]


#Batching the data to the GPU
def getBatch(split,batch_size, context_length):
    d = train_data if split=="train" else val_data
    ix = torch.randint(len(d)-context_length, (batch_size,))
    x_list = []
    y_list = []
    for i in ix:
        x_chunk = d[i:i+context_length]
        y_chunk = d[i+1:i+1+context_length]
        x_list.append(x_chunk)
        y_list.append(y_chunk)
    x = torch.stack(x_list)
    y = torch.stack(y_list)
    return x.to(device), y.to(device)

input_shape, target_shape =getBatch("train",3,8)
# print(f"Input shape will be : {decode(input_shape[0].tolist())}")
# print(f"Output shape will be : {decode(target_shape[0].tolist())}")


# RMS Normalization 
DROPOUT = 0.2
demo_x = torch.randn(2,4,8, device=device)
norm = nn.RMSNorm(8, eps=1e-6).to(device)
demo_out = norm(demo_x)

# Multi-Head Attention (MHA)
mha = nn.MultiheadAttention(
    embed_dim=256,
    num_heads=8,
    batch_first=True
).to(device)

# Built-in RoPE
rope = RotaryEmbedding(dim=32)  # 266/8 = 32

demo_in = torch.randn(2, 16, 256, device=device)

# so RoPE cannot be applied here.
attn_output, attn_weights = mha(
    demo_in,
    demo_in,
    demo_in
)

# SwiGLU - Feed Forwarding Layer
ffn = SwiGLU(
    in_features=768,
    hidden_features=3072,
    out_features=768
)