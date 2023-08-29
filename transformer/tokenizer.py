import io
import re


from collections import Counter
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import vocab


def robust(tokenize_fn):
    def robust_tokenize_fn(sentence):
        sentence = re.sub(r"[\*\"“”\n\\…\+\-\/\=\(\)‘•:\[\]\|’\!;]", " ", str(sentence))
        sentence = re.sub(r"[ ]+", " ", sentence)
        sentence = re.sub(r"\!+", "!", sentence)
        sentence = re.sub(r"\,+", ",", sentence)
        sentence = re.sub(r"\?+", "?", sentence)
        sentence = sentence.lower()
        return [tok for tok in tokenize_fn(sentence) if tok != " "]

    return robust_tokenize_fn


def build_tokenizer(lang="en"):
    return robust(get_tokenizer("spacy", lang))


def build_vocab(textfile, tokenizer):
    counter = Counter()
    with io.open(textfile) as f:
        for line in f:
            counter.update(tokenizer(line))
        v = vocab(counter, specials=["<UNKNOWN>", "<PAD>", "<SOS>", "<EOS>", "<UNK>"])
        v.set_default_index(v["<UNK>"])
        return v
