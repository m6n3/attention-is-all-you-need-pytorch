import decoder_layer
import positional_embedding
import torch.nn as nn


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size,
        embed_dim,
        feedforward_dim,
        num_layers,
        num_heads,
        max_seq_len,
        dropout,
    ):
        super().__init__()
        self.positional_embedding = positional_embedding.PositionalEmbedding(
            embed_dim=embed_dim,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            dropout=dropout,
        )
        self.decoder_layers = nn.ModuleList(
            [
                decoder_layer.DecoderLayer(
                    embed_dim=embed_dim,
                    feedforward_dim=feedforward_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                )
                for _ in range(num_layers)
            ]
        )
        self.linear = nn.Linear(embed_dim, vocab_size)  # vocab_size is output dim

    def forward(self, x, mask, enc_out, enc_mask):
        # x: [batch size, seq len]
        # mask: [batch size, 1, 1, seq len]
        # enc_out: [batch size, seq len, embed dim]
        # enc_mask: [batch size, 1, seq len, seq len]

        # Note: enc_mask.shape[1] = 1, because this mask will be applied to all
        # the heads, which will have a shape of
        # [batch size, heads, seq len, seq len].

        x = self.positional_embedding(x)
        # x: [batch size, seq len, embed dim]

        for decoder in self.decoder_layers:
            dec_in, _ = decoder(x, mask, enc_out, enc_mask)
        # x: [batch size, seq len, embed dim]

        # softmax is not needed as we are using torch.nn.CrossEntropyLoss() for loss.
        out = self.linear(x)
        # out: [batch size, seq len, vocab size]

        return out
