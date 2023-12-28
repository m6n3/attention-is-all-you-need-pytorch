import torch
import unittest

from transformer import dataset as d


class TestDataset(unittest.TestCase):
    def test_basics(self):
        dataset = d.Dataset(
            src_lang="en_core_web_sm",
            src_filepath="transformer/testdata/english.txt",
            trg_lang="fr_core_news_sm",
            trg_filepath="transformer/testdata/french.txt",
        )
        batch_size = 2
        src_vocab, trg_vocab = dataset.get_src_vocab(), dataset.get_trg_vocab()

        dataloader = d.build_dataloader(
            dataset=dataset,
            src_vocab=src_vocab,
            trg_vocab=trg_vocab,
            batch_size=batch_size,
            device=torch.device("cpu"),
        )

        src_sos, src_eos, src_pad = (
            src_vocab["<SOS>"],
            src_vocab["<EOS>"],
            src_vocab["<PAD>"],
        )
        trg_sos, trg_eos, trg_pad = (
            trg_vocab["<SOS>"],
            trg_vocab["<EOS>"],
            trg_vocab["<PAD>"],
        )

        for _, (src_batch, trg_batch) in enumerate(dataloader):
            src_batch = src_batch.permute(1, 0)
            trg_batch = trg_batch.permute(1, 0)

            self.assertEqual(src_batch.shape[0], batch_size)
            self.assertEqual(trg_batch.shape[0], batch_size)

            for _, src_tensor in enumerate(src_batch):
                self.assertTrue(src_tensor[0] == src_sos)
                self.assertTrue(src_tensor[-1] in [src_eos, src_pad])

            for _, trg_tensor in enumerate(trg_batch):
                self.assertTrue(trg_tensor[0] == trg_sos)
                self.assertTrue(trg_tensor[-1] in [trg_eos, trg_pad])


if __name__ == "__main__":
    unittest.main()
