from residual import Residual

import torch
import unittest


class TestResidual(unittest.TestCase):
    def test_basics(self):
        embed_dim, dropout = 512, 0.1
        res = Residual(embed_dim, dropout)

        batch_size, seq_len = 10, 50
        x = torch.rand(batch_size, seq_len, embed_dim)
        r = torch.rand(batch_size, seq_len, embed_dim)

        out = res(x, r)
        self.assertEqual(out.shape, torch.Size([batch_size, seq_len, embed_dim]))


if __name__ == "__main__":
    unittest.main()
