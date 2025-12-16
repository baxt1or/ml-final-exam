import numpy as np
import torch
from tokenizers import Tokenizer

from model.transformer import EncoderTransformerClassifier

# Load tokenizer
tokenizer = Tokenizer.from_file("./model/tokenizer.json")

# Load model
model = EncoderTransformerClassifier()


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
print("Model:", 'positive' if out == 2 else 'neutral' if out == 1 else 'negative')
