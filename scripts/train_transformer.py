import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
import regex as re
import warnings
import swifter

from torch.utils.data import TensorDataset, DataLoader


warnings.filterwarnings("ignore")



vocab_size=30000

batch_size = 64
block_size = 80 # number of tokens in one sentence
learning_rate = 3e-3

max_iter = 10

n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.2


text_column = "clean_text"
target_column = "rnk"



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


def normalize_uzum_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    cleans the text based on the following criterias listed below
    :param df: pandas dataframe
    :returns: cleaned pandas dataframe
    """

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
        # skip unwanted characters
            if ch in final_clean:
               continue

        # keep only allowed characters (latin, cyrillic, digits, spaces)
            if not allowed_re.fullmatch(ch):
               continue

        # map special latin → uzbek letters
            out.append(latin_map.get(ch, ch))

        return "".join(out)


    df['clean_text'] = df["normalized_review_text"].astype(str).swifter.apply(normalize_text)

    return df




def get_token_data():
    """
    Trains a BPE tokenizer on the text column, encodes + pads the texts
    :returns: X (tensor), y (tensor), tokenizer (trained)
    """

    uzum_df = normalize_uzum_reviews(uzum_reviews_filtered_df)


    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()


    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<pad>", "<unk>"]
    )


    tokenizer.train_from_iterator(uzum_df[text_column].astype(str).tolist(), trainer)

    PAD_ID = tokenizer.token_to_id("<pad>")
    UNK_ID = tokenizer.token_to_id("<unk>")


    def padding_sentence(ids, max_len=block_size, pad_id=PAD_ID):
        if len(ids) < max_len:
            ids += [pad_id] * (max_len - len(ids))
        return ids[:max_len]


    X_seq = [padding_sentence(tokenizer.encode(str(t)).ids) for t in uzum_df[text_column]]


    X = torch.tensor(X_seq, dtype=torch.long)
    y = torch.tensor(uzum_df[target_column].values, dtype=torch.long)

    return X, y, tokenizer




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




class EncoderTransformerClassifier(nn.Module):
    def __init__(self, vocab_size, n_embd, n_head, n_layer, block_size, n_classes):
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
X, y, tokenizer = get_token_data()

dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=64, shuffle=True)



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using:", device)


model = EncoderTransformerClassifier(
    vocab_size=vocab_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    block_size=block_size,
    n_classes=3
).to(device)



optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


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



probs = model.predict("manga yoqdi mol", tokenizer)[0].tolist()

out = np.argmax(probs)

print(probs)
print('excellent' if out == 2 else 'fair' if out == 1 else 'poor')



# Save model state_dict
torch.save(model.state_dict(), "encoder_transformer_classifier.pth")

# Save tokenizer
tokenizer.save("tokenizer.json")
