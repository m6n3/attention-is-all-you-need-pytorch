from transformer import dataset, trainer, transformer

import torch
import unittest


class TestTrainer(unittest.TestCase):
    def test_train(self):
        d = dataset.Dataset(
            src_lang="en_core_web_sm",
            src_filepath="transformer/testdata/english.txt",
            trg_lang="fr_core_news_sm",
            trg_filepath="transformer/testdata/french.txt",
        )
        model = transformer.Transformer(
            src_vocab=d.get_src_vocab(),
            trg_vocab=d.get_trg_vocab(),
            num_enc_layers=2,
            num_dec_layers=2,
        )
        t = trainer.Trainer(
            model=model, dataset=d, train_batch_size=2, save_folder=""
        )  # Do not save the model.

        t.train()


if __name__ == "__main__":
    unittest.main()
