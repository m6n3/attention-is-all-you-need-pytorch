import math
import torch
import torch.nn as nn
from torch.autograd import Variable


class PositionalEmbedding(nn.Module):
    def __init__(self, embed_dim, vocab_size, max_seq_len, dropout, device) -> None:
        super().__init__()
        assert embed_dim % 2 == 0

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.pe = self.build_pe_matrix(max_seq_len, embed_dim)
        self.scale = math.sqrt(embed_dim)
        self.device = device

    def build_pe_matrix(self, max_seq_len, embed_dim):
        pe = torch.zeros(max_seq_len, embed_dim)
        for pos in range(max_seq_len):
            for i in range(0, embed_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (i / embed_dim)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** (i / embed_dim)))
        # pe: [max seq len, embed dim]

        return pe

    def forward(self, src):
        # src: [batch size, seq len]

        embed = self.embedding(src)
        # embed: [batch size, seq len, embed dim]

        seq_len = src.size(1)
        positional_encoding = Variable(self.pe[:seq_len, :], requires_grad=False).to(
            self.device
        )
        # positional_encoding: [seq len, embed size]

        positional_embedding = (
            embed * self.scale + positional_encoding
        )  # `+` takes care of size mismatch of the operands.
        # positional_embedding: [batch size, seq len, embed dim]

        return self.dropout(positional_embedding)
