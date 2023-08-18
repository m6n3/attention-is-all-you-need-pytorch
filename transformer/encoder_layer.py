from attention import MultiHeadAttention
from residual import Residual
from feedforward import FeedForward

import torch
import torch.nn as nn


class EncoderLayer(nn.Module):
    def __init__(self, embed_dim, feedforward_dim, num_heads, dropout) -> None:
        super().__init__()
        self.multi_head_attention = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.residual_1 = Residual(embed_dim, dropout)
        self.feed_forward = FeedForward(embed_dim, feedforward_dim, dropout)
        self.residual_2 = Residual(embed_dim, dropout)

    def forward(self, x, mask=None):
        # x: [batch size, seq len, embed dim]
        # mask: [batch size, 1, 1, seq len]

        attention_out, _ = self.multi_head_attention(x, x, x, mask)
        # attention_out: [batch size, seq len, embed dim]

        residul_out_1 = self.residual_1(attention_out, x)
        # residual_out_1: [batch size, seq len, embed dim]

        feedforward_out = self.feed_forward(residul_out_1)
        # feedforward_out: [batch size, seq len, embed dim]

        out = self.residual_2(feedforward_out, residul_out_1)
        # out: [batch size, seq len, embed dim]

        return out
