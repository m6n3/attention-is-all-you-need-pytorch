import encoder_layer
import torch
import unittest


class TestEncoderLayer(unittest.TestCase):
    def test_basics(self):
        embed_dim, feedforward_dim, num_heads, dropout = 512, 2048, 8, 0.1
        el = encoder_layer.EncoderLayer(embed_dim, feedforward_dim, num_heads, dropout)
        batch_size, seq_len = 50, 100
        x = torch.rand(batch_size, seq_len, embed_dim)
        mask = torch.rand(batch_size, 1, 1, seq_len)

        y = el(x, mask)

        self.assertEqual(y.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
