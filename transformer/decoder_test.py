from transformer import decoder

import torch
import unittest


class TestDecoder(unittest.TestCase):
    def test_basics(self):
        vocab_size, embed_dim = 10_000, 512
        d = decoder.Decoder(
            vocab_size=vocab_size,
            embed_dim=embed_dim,
            feedforward_dim=2048,
            num_layers=2,
            num_heads=8,
            max_seq_len=100,
            dropout=0.1,
            device=torch.device("cpu"),
        )

        batch_size, seq_len = 50, 100
        x = torch.randint(vocab_size, (batch_size, seq_len))
        mask = torch.rand(batch_size, 1, 1, seq_len) > 0.5
        enc_out = torch.rand(batch_size, seq_len, embed_dim) * 10
        enc_mask = torch.rand(batch_size, 1, seq_len, seq_len)

        y = d(x, mask, enc_out, enc_mask)
        self.assertEqual(y.shape, torch.Size([batch_size, seq_len, vocab_size]))


if __name__ == "__main__":
    unittest.main()
