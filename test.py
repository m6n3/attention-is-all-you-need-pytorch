import torch

from transformer import dataset, tokenizer, transformer, trainer, translator

d = dataset.Dataset(
    src_lang="en_core_web_sm",  # see spacy.io
    src_filepath="./transformer/testdata/english.txt",
    trg_lang="fr_core_news_sm",
    trg_filepath='./transformer/testdata/french.txt'
)

model = transformer.Transformer(
    src_vocab=d.get_src_vocab(),
    trg_vocab=d.get_trg_vocab(),
    hid_dim=512,
    feedforward_dim=2048,
    num_enc_layers=6,
    num_dec_layers=6,
    num_attention_heads=8,
    max_seq_len=100,
    dropout=0.1,
)

tr = trainer.Trainer(
  model=model,
  dataset=d,
  train_batch_size=64,
  train_lr=8e-5,
  train_epochs=4,
  train_num_steps=10,  # Training on a single batch of data is considered one step
  save_every_n_steps=100,
  save_folder="."
)

tr.train()

t = translator.Translator(
    model=model,
    src_vocab=d.get_src_vocab(),
    trg_vocab=d.get_trg_vocab(),
    src_tokenizer=tokenizer.build_tokenizer(lang="en_core_web_sm"),
    max_trg_sentence_len=100,
)


translated = t.translate("a sentence in en.")

