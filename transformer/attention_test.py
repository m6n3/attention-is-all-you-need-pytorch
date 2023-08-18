import attention as a

import torch
import torch.nn as nn
import unittest


class TestAttention(unittest.TestCase):
    def test_basics(self):
        embed_dim, num_heads, dropout = 512, 8, 0.1
        m = a.MultiHeadAttention(embed_dim, num_heads, dropout)
        batch_size, seq_len = 50, 100
        x = torch.rand(batch_size, seq_len, embed_dim)
        mask = torch.rand(batch_size, 1, 1, seq_len) > 0.5
        y, attn = m(x, x, x, mask)

        self.assertEqual(y.shape, x.shape)
        self.assertEqual(
            attn.shape, torch.Size([batch_size, num_heads, seq_len, seq_len])
        )


if __name__ == "__main__":
    unittest.main()
