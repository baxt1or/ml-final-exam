import pandas as pd
import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim


vocab_size=30000

batch_size = 64
block_size = 80 # number of tokens in one sentence
learning_rate = 3e-4


max_iters = 5000
eval_iters = 200
eval_interval = 500


n_embd = 84
n_head = 4
n_layer = 4 
dropout = 0.2


X, y, tokenizer = get_token_data()