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

