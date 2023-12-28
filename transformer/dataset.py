import io
import torch
import urllib.parse

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from transformer import tokenizer


class Dataset(torch.utils.data.Dataset):
    """Build a dataset from src and trg files.

    It expects each line in destination file be the translation of its
    correspoinding line in src file.
    """

    def __init__(self, src_lang, src_filepath, trg_lang, trg_filepath):
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.src_file = src_filepath
        self.trg_file = trg_filepath

        # Build tokenizers and vocabs
        self.src_tokenizer = tokenizer.build_tokenizer(lang=src_lang)
        self.trg_tokenizer = tokenizer.build_tokenizer(lang=trg_lang)
        self.src_vocab = tokenizer.build_vocab(self.src_file, self.src_tokenizer)
        self.trg_vocab = tokenizer.build_vocab(self.trg_file, self.trg_tokenizer)

        # Convert texts to tensors.
        with io.open(self.src_file, encoding="utf8") as src_f, io.open(
            self.trg_file, encoding="utf8"
        ) as trg_f:
            self.tensors = []
            for src_sentence, trg_sentence in zip(iter(src_f), iter(trg_f)):
                src_tensor = torch.tensor(
                    [self.src_vocab[tok] for tok in self.src_tokenizer(src_sentence)]
                )
                trg_tensor = torch.tensor(
                    [self.trg_vocab[tok] for tok in self.trg_tokenizer(trg_sentence)]
                )
                self.tensors.append((src_tensor, trg_tensor))

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, i):
        return self.tensors[i]

    def get_src_vocab(self):
        return self.src_vocab

    def get_trg_vocab(self):
        return self.trg_vocab


def build_dataloader(dataset, src_vocab, trg_vocab, batch_size, device):
    def batcher(data):
        src_sos, src_eos, src_pad = (
            src_vocab["<SOS>"],
            src_vocab["<EOS>"],
            src_vocab["<PAD>"],
        )
        trg_sos, trg_eos, trg_pad = (
            trg_vocab["<SOS>"],
            trg_vocab["<EOS>"],
            trg_vocab["<PAD>"],
        )

        src_batch, trg_batch = [], []
        for src_tensor, trg_tensor in data:
            src_batch.append(
                torch.cat(
                    [torch.tensor([src_sos]), src_tensor, torch.tensor([src_eos])]
                ).to(device)
            )
            trg_batch.append(
                torch.cat(
                    [torch.tensor([trg_sos]), trg_tensor, torch.tensor([trg_eos])]
                ).to(device)
            )

        # Make batch elements of same size.
        src_batch = pad_sequence(src_batch, padding_value=src_pad)
        trg_batch = pad_sequence(trg_batch, padding_value=trg_pad)

        # src_batch: [src max(seq_len) + 2, batch_size]
        # trg_batch: [trg max(seq_len) + 2, batch_size]

        return src_batch, trg_batch

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=batcher)
