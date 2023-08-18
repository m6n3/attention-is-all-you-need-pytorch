import feedforward as ff
import unittest
import torch


class TestFeedForward(unittest.TestCase):
    def test_basics(self):
        embed_dim, feedforward_dim, dropout = 512, 2048, 0.1
        m = ff.FeedForward(embed_dim, feedforward_dim, dropout)

        batch_size, seq_len = 50, 100
        x = torch.rand(batch_size, seq_len, embed_dim)
        y = m(x)

        self.assertEqual(y.shape, x.shape)


if __name__ == "__main__":
    unittest.main()
