from transformer import encoder_layer
from transformer import positional_embedding

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(
        self,
        *,
        vocab_size,
        embed_dim,
        feedforward_dim,
        num_layers,
        num_heads,
        max_seq_len,
        dropout,
        device,
    ):
        super().__init__()
        self.positional_embedding = positional_embedding.PositionalEmbedding(
            embed_dim=embed_dim,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            dropout=dropout,
            device=device,
        )
        self.encoder_layers = nn.ModuleList(
            [
                encoder_layer.EncoderLayer(
                    embed_dim=embed_dim,
                    feedforward_dim=feedforward_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask):
        # x: [batch size, seq len]

        x = self.positional_embedding(x)
        # x: [batch size, seq len, embed dim]

        for layer in self.encoder_layers:
            x = layer(x, mask)
        # x: [batch size, seq len, embed dim]

        return x
