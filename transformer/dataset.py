import io
import torch
import urllib.parse

from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

import tokenizer


class Dataset(torch.utils.data.Dataset):
    """Build a dataset from src and dst files.

    It expects each line in destination file be the translation of its
    correspoinding line in src file.
    """

    def __init__(self, src_lang, src_filepath, dst_lang, dst_filepath):
        self.src_lang = src_lang
        self.dst_lang = dst_lang
        self.src_file = src_filepath
        self.dst_file = dst_filepath

        # Build tokenizers and vocabs
        self.src_tokenizer = tokenizer.build_tokenizer(lang=src_lang)
        self.dst_tokenizer = tokenizer.build_tokenizer(lang=dst_lang)
        self.src_vocab = tokenizer.build_vocab(self.src_file, self.src_tokenizer)
        self.dst_vocab = tokenizer.build_vocab(self.dst_file, self.dst_tokenizer)

        # Convert texts to tensors.
        with io.open(self.src_file, encoding="utf8") as src_f, io.open(
            self.dst_file, encoding="utf8"
        ) as dst_f:
            self.tensors = []
            for src_sentence, dst_sentence in zip(iter(src_f), iter(dst_f)):
                src_tensor = torch.tensor(
                    [self.src_vocab[tok] for tok in self.src_tokenizer(src_sentence)]
                )
                dst_tensor = torch.tensor(
                    [self.dst_vocab[tok] for tok in self.dst_tokenizer(dst_sentence)]
                )
                self.tensors.append((src_tensor, dst_tensor))

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, i):
        return self.tensors[i]

    def get_src_vocab(self):
        return self.src_vocab

    def get_dst_vocab(self):
        return self.dst_vocab


def build_dataloader(dataset, src_vocab, dst_vocab, batch_size):
    def batcher(data):
        src_sos, src_eos, src_pad = (
            src_vocab["<SOS>"],
            src_vocab["<EOS>"],
            src_vocab["<PAD>"],
        )
        dst_sos, dst_eos, dst_pad = (
            dst_vocab["<SOS>"],
            dst_vocab["<EOS>"],
            dst_vocab["<PAD>"],
        )

        src_batch, dst_batch = [], []
        for src_tensor, dst_tensor in data:
            src_batch.append(
                torch.cat(
                    [torch.tensor([src_sos]), src_tensor, torch.tensor([src_eos])]
                )
            )
            dst_batch.append(
                torch.cat(
                    [torch.tensor([dst_sos]), dst_tensor, torch.tensor([dst_eos])]
                )
            )

        # Make batch elements of same size.
        src_batch = pad_sequence(src_batch, padding_value=src_pad)
        dst_batch = pad_sequence(dst_batch, padding_value=dst_pad)

        # src_batch: [src max(seq_len) + 2, batch_size]
        # dst_batch: [dst max(seq_len) + 2, batch_size]

        return src_batch, dst_batch

    return DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=batcher)
