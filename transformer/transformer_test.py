from transformer import transformer, tokenizer

import torch
import unittest


class TestTransformer(unittest.TestCase):
    def test_basics(self):
        src_vocab = tokenizer.build_vocab(
            "transformer/testdata/english.txt",
            tokenizer.build_tokenizer("en_core_web_sm"),
        )
        trg_vocab = tokenizer.build_vocab(
            "transformer/testdata/french.txt",
            tokenizer.build_tokenizer("fr_core_news_sm"),
        )
        t = transformer.Transformer(
            src_vocab=src_vocab, trg_vocab=trg_vocab, hid_dim=512
        )

        # vocab_size is set to 2 as we have a very small src and trg test files with
        # few words.
        batch_size, seq_len, vocab_size = 50, 100, 2
        src = torch.randint(vocab_size, (batch_size, seq_len))
        trg = torch.randint(vocab_size, (batch_size, seq_len))
        out = t(src, trg)

        self.assertEqual(out.shape, torch.Size([batch_size, seq_len, len(trg_vocab)]))


if __name__ == "__main__":
    unittest.main()
