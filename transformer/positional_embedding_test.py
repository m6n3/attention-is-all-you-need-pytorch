from transformer import positional_embedding as pe

import torch
import unittest


class TestPositionalEmbedding(unittest.TestCase):
    def test_basics(self):
        embed_dim, vocab_size, max_seq_len, dropout = 512, 1_000_000, 100, 0.1

        embedder = pe.PositionalEmbedding(
            embed_dim, vocab_size, max_seq_len, dropout, device=torch.device("cpu")
        )

        batch_size, seq_len = 5, 50
        x = torch.randint(vocab_size, (batch_size, seq_len))
        out = embedder(x)

        self.assertEqual(out.shape, torch.Size([batch_size, seq_len, embed_dim]))


if __name__ == "__main__":
    unittest.main()
