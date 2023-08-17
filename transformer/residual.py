import torch.nn as nn


class Residual(nn.Module):
    def __init__(self, embed_dim, dropout):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, residual):
        return self.norm(self.dropout(input) + residual)
