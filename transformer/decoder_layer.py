from attention import MultiHeadAttention
from residual import Residual
from feedforward import FeedForward

import torch
import torch.nn as nn


class DecoderLayer(nn.Module):
    def __init__(self, embed_dim, feedforward_dim, num_heads, dropout) -> None:
        super().__init__()
        self.multi_head_attention_1 = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.residual_1 = Residual(embed_dim, dropout)
        self.multi_head_attention_2 = MultiHeadAttention(embed_dim, num_heads, dropout)
        self.residual_2 = Residual(embed_dim, dropout)
        self.feed_forward = FeedForward(embed_dim, feedforward_dim, dropout)
        self.residual_3 = Residual(embed_dim, dropout)

    def forward(self, x, mask, enc_out, enc_mask):
        # x: [batch size, seq len, embed dim]
        # mask: [batch size, 1, 1, seq len]
        # enc_out: [batch size, seq len, embed dim]
        # enc_mask: [batch size, 1, seq len, seq len]

        x_attn, _ = self.multi_head_attention_1(x, x, x, mask)
        # x_attn: [batch size, seq len, embed dim]

        x_res = self.residual_1(x_attn, x)
        # x_res: [batch size, seq len, embed dim]

        # attending encoder's output
        x_attn, enc_attention_coeff = self.multi_head_attention_2(
            x_res, enc_out, enc_out, enc_mask
        )
        # x_attn: [batch size, seq len, embed dim]
        # enc_attention_coeff = [batch size, seq len, seq len]

        x_res = self.residual_2(x_attn, x_res)
        # x_res: [batch size, seq len, embed dim]

        x_ff = self.feed_forward(x_res)
        # x_ff: [batch size, seq len, embed dim]

        x_res = self.residual_3(x_ff, x_res)
        # x_res: [batch size, seq len, embed dim]

        return x_res, enc_attention_coeff
