# -*-config:utf-8 -* 
import pandas as pd

letters = " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZàéèêùîç()\'\",.-?:;!"

fr_symbols = {
    "+" : "plus",
    "=" : "egal",
    "%": "pourcent",
    "$": "dollar",
    "€": "euro",
    "ï": "i",
    "ë": "e",
    "ü": "u",
}

def fr_cleaner(phrase, lower=False):
    if lower: phrase = phrase.lower()
    for symbol, string in fr_symbols.items():
        phrase = phrase.replace(symbol, string)
    return phrase

def get_vocab(min_code=3):
    index_df = [c for c in letters]
    col_df = ['caractere', 'numero']
    vocab = [[c, i+min_code] for i, c in enumerate(letters)]
    return pd.DataFrame(vocab, index=index_df, columns=col_df)

