import encoder
import torch
import unittest


class TestEncoder(unittest.TestCase):
    def test_basics(self):
        with torch.no_grad():
            vocab_size, embed_dim = 1000_000, 512
            e = encoder.Encoder(
                vocab_size=vocab_size,
                embed_dim=512,
                feedforward_dim=2048,
                num_layers=1,
                num_heads=8,
                max_seq_len=100,
                dropout=0.1,
            )
            batch_size, seq_len = 50, 100
            x = torch.randint(vocab_size, (batch_size, seq_len))
            mask = torch.rand(batch_size, 1, 1, seq_len) > 0.5

            y = e(x, mask)
            self.assertEqual(y.shape, torch.Size([batch_size, seq_len, embed_dim]))


if __name__ == "__main__":
    unittest.main()
