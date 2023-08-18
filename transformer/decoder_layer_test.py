import decoder_layer
import torch
import unittest


class TestDecoderLayer(unittest.TestCase):
    def test_basics(self):
        embed_dim, num_heads = 512, 8
        dl = decoder_layer.DecoderLayer(
            embed_dim=embed_dim, feedforward_dim=2048, num_heads=num_heads, dropout=0.1
        )
        batch_size, seq_len = 50, 100
        x = torch.rand(batch_size, seq_len, embed_dim)
        mask = torch.rand(batch_size, 1, 1, seq_len) > 0.5
        enc_out = torch.rand(batch_size, seq_len, embed_dim)
        enc_mask = torch.rand(batch_size, 1, seq_len, seq_len) > 0.5

        y, attn_coeff = dl(x, mask, enc_out, enc_mask)

        self.assertEqual(y.shape, x.shape)
        self.assertEqual(
            attn_coeff.shape, torch.Size([batch_size, num_heads, seq_len, seq_len])
        )


if __name__ == "__main__":
    unittest.main()
