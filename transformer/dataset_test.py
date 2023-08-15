import torch
import unittest

import dataset as d

# from dataset import Dataset


class TestDataset(unittest.TestCase):
    def test_basics(self):
        dataset = d.Dataset(
            src_lang="en_core_web_sm",
            src_filepath="./testdata/english.txt",
            dst_lang="fr_core_news_sm",
            dst_filepath="./testdata/french.txt",
        )
        batch_size = 2
        src_vocab, dst_vocab = dataset.get_src_vocab(), dataset.get_dst_vocab()

        dataloader = d.build_dataloader(
            dataset=dataset,
            src_vocab=src_vocab,
            dst_vocab=dst_vocab,
            batch_size=batch_size,
        )

        src_sos, src_eos, src_pad = (
            src_vocab["<SOS>"],
            src_vocab["<EOS>"],
            src_vocab["<PAD>"],
        )
        dst_sos, dst_eos, dst_pad = (
            dst_vocab["<SOS>"],
            dst_vocab["<EOS>"],
            dst_vocab["<PAD>"],
        )

        for _, (src_batch, dst_batch) in enumerate(dataloader):
            src_batch = src_batch.permute(1, 0)
            dst_batch = dst_batch.permute(1, 0)

            self.assertEqual(src_batch.shape[0], batch_size)
            self.assertEqual(dst_batch.shape[0], batch_size)

            for _, src_tensor in enumerate(src_batch):
                self.assertTrue(src_tensor[0] == src_sos)
                self.assertTrue(src_tensor[-1] in [src_eos, src_pad])

            for _, dst_tensor in enumerate(dst_batch):
                self.assertTrue(dst_tensor[0] == dst_sos)
                self.assertTrue(dst_tensor[-1] in [dst_eos, dst_pad])


if __name__ == "__main__":
    unittest.main()
