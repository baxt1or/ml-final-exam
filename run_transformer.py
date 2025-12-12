import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizers import Tokenizer

from model.transformer import EncoderTransformerClassifier, vocab_size, n_embd, n_head, n_layer, block_size


# Load tokenizer
tokenizer = Tokenizer.from_file("./model/tokenizer.json")

# Load model
model = EncoderTransformerClassifier(
    vocab_size=vocab_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    block_size=block_size,
    n_classes=3
)

model.load_state_dict(torch.load("./model/encoder_transformer_classifier.pth", map_location='cpu'))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()


# getting a user text review
text = input("Enter your comments: ")

probs = model.predict(text, tokenizer)[0]  
out = np.argmax(probs)

print("You wrote: ", text)
print("Class probabilities: ", probs)
print("Model:", 'excellent' if out == 2 else 'fair' if out == 1 else 'poor')
