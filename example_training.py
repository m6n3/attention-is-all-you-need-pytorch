import os
import torch

from transformer import dataset, tokenizer, transformer, trainer, translator

SRC_LANG = "en_core_web_sm"
SRC_TRAINING_FILE = "./transformer/data/english.txt"
TARGET_LANG = "fr_core_news_sm"
TARGET_TRAINING_FILE = "./transformer/data/french.txt"
CHECKPOINT_PATH = "./checkpoint.pt"


if __name__ == "__main__":
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

  tr = trainer.Trainer(
      model=model,
      dataset=d,
      train_batch_size=64,
      train_lr=8e-5,
      train_epochs=4,
      train_num_steps=10,  # Training on a single batch of data is considered one step
      checkpoint_every_n_steps=100,
      checkpoint_path=CHECKPOINT_PATH,
      use_gpu=True,
  )

  # Start from last checkpoint (if any).
  if os.path.exists(CHECKPOINT_PATH):
      tr.load_checkpoint(CHECKPOINT_PATH)

  tr.train()
