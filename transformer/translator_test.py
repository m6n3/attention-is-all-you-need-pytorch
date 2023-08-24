import dataset
import tokenizer
import transformer
import translator

import unittest


class TestTranslator(unittest.TestCase):
    def test_translate(self):
        d = dataset.Dataset(
            src_lang="en_core_web_sm",
            src_filepath="./testdata/english.txt",
            dst_lang="fr_core_news_sm",
            dst_filepath="./testdata/french.txt",
        )
        model = transformer.Transformer(
            src_vocab=d.get_src_vocab(),
            trg_vocab=d.get_dst_vocab(),
            num_enc_layers=2,
            num_dec_layers=2,
        )
        t = translator.Translator(
            model=model,
            src_vocab=d.get_src_vocab(),
            trg_vocab=d.get_dst_vocab(),
            src_tokenizer=tokenizer.build_tokenizer(lang="fr_core_news_sm"),
            max_trg_sentence_len=5,
        )

        # sentence from dataset, so src_vocab has it.
        # testing for no crash.
        res = t.translate("Go.")


if __name__ == "__main__":
    unittest.main()
