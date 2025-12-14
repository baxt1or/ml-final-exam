import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import regex as re
import warnings
import swifter
from torch.utils.data import TensorDataset, DataLoader
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
warnings.filterwarnings("ignore")


# ALL PARAMETERS
vocab_size=30000
batch_size = 64
block_size = 80 
learning_rate = 3e-3

max_iter = 10

n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.2

n_classes = 3



# PREPROCESSING UZUM TEXT DATA
def get_normalized_uzum_reviews():
    """
    cleans the text based on the following criterias listed below
    :param df: pandas dataframe
    :returns: cleaned pandas dataframe
    """

    uzum_reviews_df = pd.read_parquet("./data/uzum_dataset.parquet", engine='pyarrow')
    uzum_reviews_df["len"] = uzum_reviews_df["normalized_review_text"].str.len()
    uzum_reviews_filtered_df = uzum_reviews_df[uzum_reviews_df["len"] <= block_size]
    rating_map = {
    'very poor' : 0,
    'poor' : 0,
    'fair' : 1,
    'good' : 2,
    'excellent' : 2
     }

    uzum_reviews_filtered_df["rnk"] = uzum_reviews_filtered_df["rating"].map(rating_map)


    latin = r"\p{Latin}"
    cyrillic = r"\p{Cyrillic}"
    digits = r"\p{Number}"


    allowed_re = re.compile(fr"(?:{latin}|{cyrillic}|{digits}|\s)")

    final_clean = {'ø','ʔ','ʕ','ʖ','ᴥ','ᵕ','⅚','ᴗ'}

    latin_map = {
    "à": "a", "á": "a", "â": "a", "ã": "a",
    "ç": "c",
    "è": "e", "é": "e", "ë": "e",
    "ì": "i", "í": "i",
    "ñ": "n",
    "ò": "o", "ó": "o", "ô": "o", "õ": "o", "ö": "o",
    "ù": "u", "ú": "u", "û": "u", "ü": "u",
    "ý": "y", "ÿ": "y",
    "ĝ": "g'", "ğ": "g'", "ġ": "g'", "ģ": "g'",
    "ĥ": "h",
    "ı": "i",
    "ĵ": "j",
    "ķ": "k",
    "ĺ": "l", "ļ": "l",
    "ń": "n", "ň": "n",
    "ō": "o'", "ŏ": "o'", "ő": "o'",
    "ŕ": "r",
    "ś": "s", "ş": "sh",
    "ũ": "u", "ū": "u", "ů": "u",
    "ź": "z", "ž": "j",
    "ǒ": "o'", "ǫ": "q",
    "ǵ": "g'",
    "ɓ": "b",
    "ə": "e",
    '²': '2',
    '³': '3',
    '¹': '1',
    'ď': 'd',
    'ɢ': 'g',
    'ɪ': 'i',
    'ɴ': 'n',
    'ʀ': 'r',
    'ʏ': 'y',
    'ʜ': 'h',
    'ʟ': 'l',
    'ө': 'o',
    'ᴀ': 'a',
    'ᴄ': 'c',
    'ᴅ': 'd',
    'ᴇ': 'e',
    'ᴊ': 'j',
    'ᴋ': 'k',
    'ᴍ': 'm',
    'ᴏ': 'o',
    'ᴘ': 'p',
    'ᴛ': 't',
    'ᴜ': 'u',
    '⁰': '0',
    '⁴': '4',
    '⁵': '5'
}


    def normalize_text(text: str) -> str:
        out = []
        for ch in text:
            # skips unnessary ones
            if ch in final_clean:
               continue

            # keeps only necessary chars
            if not allowed_re.fullmatch(ch):
               continue

            # maps final
            out.append(latin_map.get(ch, ch))

        return "".join(out)


    uzum_reviews_filtered_df['clean_text'] = uzum_reviews_filtered_df["normalized_review_text"].astype(str).swifter.apply(normalize_text)

    return uzum_reviews_filtered_df[['clean_text', 'rnk']]


# TOKENIZES DATA 
def get_token_data():
    """
    trains a BPE tokenizer on the text column, encodes + pads the texts
    :returns: X (tensor), y (tensor), tokenizer (trained)
    """

    uzum_df = get_normalized_uzum_reviews()


    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()


    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>"]
    )


    tokenizer.train_from_iterator(uzum_df["clean_text"].astype(str).tolist(), trainer)

    PAD_ID = tokenizer.token_to_id("<pad>")
    UNK_ID = tokenizer.token_to_id("<unk>")


    def padding_sentence(ids):
        if len(ids) < block_size:
            ids += [PAD_ID] * (block_size - len(ids))
        return ids[:block_size]


    X_seq = [padding_sentence(tokenizer.encode(str(t)).ids) for t in uzum_df["clean_text"]]


    X = torch.tensor(X_seq, dtype=torch.long)
    y = torch.tensor(uzum_df["rnk"].values, dtype=torch.long)

    return X, y, tokenizer






# SELF-ATTENTION BLOCK
class SelfAttention(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(x)           # (B,T,hs)
        q = self.query(x)         # (B,T,hs)

        wei = q @ k.transpose(-2, -1) * C**-0.5
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v

        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList(SelfAttention(head_size) for _ in range(num_heads))
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out



class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)



class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffw = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffw(self.ln2(x))
        return x



# ENCODER ONLY TRANSFORMER
class EncoderTransformerClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.classifier = nn.Linear(n_embd, n_classes)

    def forward(self, idx, targets=None):
        idx = idx.to(self.token_embedding_table.weight.device)
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)
        pos = torch.arange(T, device=idx.device)
        pos_emb = self.position_embedding_table(pos)
        x = tok_emb + pos_emb

        x = self.blocks(x)
        x = self.ln_f(x)
        x_cls = x.mean(dim=1)

        logits = self.classifier(x_cls)

        if targets is None:
            return logits

        loss = F.cross_entropy(logits, targets)
        return logits, loss


    def predict(self, text, tokenizer):
        self.eval()

        ids = tokenizer.encode(text).ids
        if len(ids) < block_size:
            ids += [0] * (block_size - len(ids))
        ids = ids[:block_size]

        x = torch.tensor([ids], dtype=torch.long)
        x = x.to(next(self.parameters()).device)

        with torch.no_grad():
            logits = self(x)
            probs = F.softmax(logits, dim=1)

        return probs.cpu().numpy()
    



# getting data in proper format
data, targets, tokenizer = get_token_data()

dataset = TensorDataset(data, targets)
loader = DataLoader(dataset, batch_size, shuffle=True)


# transforming to cude GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


# MODEL BUILDING
model = EncoderTransformerClassifier().to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


# TRANING LOOP
for epoch in range(max_iter):
    model.train()
    total_loss = 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        logits, loss = model(xb, yb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch+1}: loss = {total_loss/len(loader):.4f}")




# getting some predictions
probs = model.predict("manga yoqdi mol", tokenizer)[0].tolist()
out = np.argmax(probs)
print(probs)
print('excellent' if out == 2 else 'fair' if out == 1 else 'poor')



# SAVING MODEL AND TOKENIZER
torch.save(model.state_dict(), "encoder_transformer_classifier.pth")
tokenizer.save("tokenizer.json")
