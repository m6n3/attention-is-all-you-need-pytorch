import os
import sys
import torch

from transformer import dataset, tokenizer, transformer, trainer, translator

SRC_LANG = "en_core_web_sm"
SRC_TRAINING_FILE = "./transformer/data/english.txt"
TARGET_LANG = "fr_core_news_sm"
TARGET_TRAINING_FILE = "./transformer/data/french.txt"
CHECKPOINT_PATH = "./checkpoint.pt"  # or download one from https://huggingface.co/m6n3xx/attention-is-all-you-need-pytorch/blob/main/checkpoint_12282023_loss_3.3954.pt


if __name__ == "__main__":
    # Need this for vocab only.
    d = dataset.Dataset(
        src_lang=SRC_LANG,
        src_filepath=SRC_TRAINING_FILE,
        trg_lang=TARGET_LANG,
        trg_filepath=TARGET_TRAINING_FILE,
    )

    model = transformer.Transformer(
        src_vocab=d.get_src_vocab(),
        trg_vocab=d.get_trg_vocab(),
        hid_dim=512,
        feedforward_dim=2048,
        num_enc_layers=3,
        num_dec_layers=3,
        num_attention_heads=8,
        max_seq_len=100,
        dropout=0.1,
        use_gpu=True,
    )

    model.load_checkpoint(CHECKPOINT_PATH)

    t = translator.Translator(
        model=model,
        src_vocab=d.get_src_vocab(),
        trg_vocab=d.get_trg_vocab(),
        src_tokenizer=tokenizer.build_tokenizer(lang=SRC_LANG),
        max_trg_sentence_len=100,
    )
    for line in sys.stdin:
        src = "Good job!"
        trg = t.translate(src)
        print(f"{src} --> {trg}")
