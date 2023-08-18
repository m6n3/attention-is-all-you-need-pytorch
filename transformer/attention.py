import math
import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout) -> None:
        super().__init__()
        self.linear_q = nn.Linear(embed_dim, embed_dim)
        self.linear_k = nn.Linear(embed_dim, embed_dim)
        self.linear_v = nn.Linear(embed_dim, embed_dim)
        self.linear_last = nn.Linear(embed_dim, embed_dim)

        assert embed_dim % num_heads == 0  # so we can divide them.

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.head_scale = math.sqrt(self.head_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        # query, key, value: [batch_size, seq_len, embed_dim]

        q = self.linear_q(query)
        k = self.linear_k(key)
        v = self.linear_v(value)
        # q,k,v : [batch size, seq len, embed dim]

        multi_heads_q = torch.permute(
            q.view(q.size(0), q.size(1), self.num_heads, self.head_dim), (0, 2, 1, 3)
        )
        multi_heads_k = torch.permute(
            k.view(k.size(0), k.size(1), self.num_heads, self.head_dim), (0, 2, 1, 3)
        )
        multi_heads_v = torch.permute(
            v.view(k.size(0), v.size(1), self.num_heads, self.head_dim), (0, 2, 1, 3)
        )
        # multi_heads_q/k/v: [batch size, num heads, seq len, head dim], where head dim = embed dim / num heads

        q_x_k = torch.matmul(multi_heads_q, multi_heads_k.transpose(3, 2))
        # q_x_k: [batch size, num heads, seq len, seq len]

        scaled_q_x_k = q_x_k / self.head_scale
        # scaled_q_x_k: [batch size, num heads, seq len, seq len]

        if mask is not None:
            scaled_q_x_k = scaled_q_x_k.masked_fill(mask == 0, -1e9)

        scores = torch.softmax(scaled_q_x_k, dim=-1)
        # scores: [batch size, num heads, seq len, seq len]

        attention_coefficients = self.dropout(scores)
        # attention_coefficients: [batch size, num heads, seq len, seq len]

        attention = torch.matmul(scores, multi_heads_v)
        # attention: [batch size, num heads, seq len, head dim]

        concat_out = (
            attention.transpose(1, 2)  # switch num heads and seq len dimentions.
            .contiguous()
            .view(attention.size(0), attention.size(2), self.embed_dim)
        )
        # concat_out: [batch size, seq len, embed dim]

        out = self.linear_last(concat_out)
        # linear_out: [batch size, seq len, embed dim]

        return out, attention_coefficients
