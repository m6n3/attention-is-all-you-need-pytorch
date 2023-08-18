import torch
import torch.nn as nn


class FeedForward(nn.Module):
    def __init__(self, embed_dim, feedforward_dim, dropout) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(embed_dim, feedforward_dim)
        self.linear_2 = nn.Linear(feedforward_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch size, seq len, embed dim]

        x = self.linear_1(x)
        # x: [batch size, seq len, feedforward dim]

        x = self.dropout(torch.relu(x))
        # x: [batch size, seq len, feedforward dim]

        x = self.linear_2(x)
        # x: [batch size, seq len, embed dim]

        return x
