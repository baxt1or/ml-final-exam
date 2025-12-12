import pandas as pd
import numpy as np 
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim

from tokenizers import Tokenizer, models, pre_tokenizers, trainers


max_len = 80
vocab_size=30000

text_column = "clean_text"
target_column = "rnk"

uzum_reviews_df = pd.read_parquet("./uzum_dataset.parquet", engine='pyarrow')
uzum_reviews_df["len"] = uzum_reviews_df["normalized_review_text"].str.len()
uzum_reviews_filtered_df = uzum_reviews_df[uzum_reviews_df["len"] <= max_len]
rating_map = {
    'very poor' : 1, 
    'poor' : 1, 
    'fair' : 2, 
    'good' : 3, 
    'excellent' : 3
}

uzum_reviews_filtered_df["rnk"] = uzum_reviews_filtered_df["rating"].map(rating_map)


def normalize_uzum_reviews(df: pd.DataFrame) -> pd.DataFrame:
    """
    cleans the text based on the following criterias listed below
    :param df: pandas dataframe
    :returns: cleaned pandas dataframe
    """


    import regex as re
    import swifter


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
    

    def padding_sentence(ids, max_len=max_len, pad_id=PAD_ID):
        if len(ids) < max_len:
            ids += [pad_id] * (max_len - len(ids))
        return ids[:max_len]
    

    X_seq = [padding_sentence(tokenizer.encode(str(t)).ids) for t in uzum_df[text_column]]
    

    X = torch.tensor(X_seq, dtype=torch.long)
    y = torch.tensor(uzum_df[target_column].values, dtype=torch.long)
    
    return X, y, tokenizer