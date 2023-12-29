# Attention is all you need: A Pytorch Implementation

## Usage

```python
import os
import torch

from transformer import dataset, tokenizer, transformer, trainer, translator

SRC_LANG=...              # E.g., "en_core_web_sm" see spacy.io
SRC_TRAINING_FILE=...     # E.g., "./transformer/data/english.txt"
TARGET_LANG=...           # E.g., "fr_core_news_sm"
TARGET_TRAINING_FILE=...  # E.g., "./transformer/data/french.txt"
CHECKPOINT_PATH=...       # E.g., "./checkpoint.pt" or download https://huggingface.co/m6n3xx/attention-is-all-you-need-pytorch/blob/main/checkpoint_12282023_loss_3.3954.pt if model config remains the same as below.

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
)

tr = trainer.Trainer(
  model=model,
  dataset=d,
  train_batch_size=64,
  train_lr=8e-5,
  train_epochs=4,
  train_num_steps=10_000,  # Training on a single batch of data is considered one step
  checkpoint_every_n_steps=100,
  checkpoint_path=CHECKPOINT_PATH,
)

# Start from last checkpoint (if any).
if os.path.exists(CHECKPOINT_PATH):
  tr.load_checkpoint(CHECKPOINT_PATH)

tr.train()

t = translator.Translator(
    model=model,
    src_vocab=d.get_src_vocab(),
    trg_vocab=d.get_trg_vocab(),
    src_tokenizer=tokenizer.build_tokenizer(lang=SRC_LANG),
    max_trg_sentence_len=100,
)


print(t.translate("a sentence in en."))
```

Alternatively


```bash
python3 -m venv myvenv
source venv3/bin/activate

git clone https://github.com/m6n3/attention-is-all-you-need-pytorch.git
cd attention-is-all-you-need-pytorch
python -m pip install --upgrade pip
python -m pip install --upgrade wheel
python -m pip install --upgrade .
pip install -r requirements.txt

python -m spacy download fr_core_news_sm
python -m spacy download en_core_web_sm

# Run all tests
nose2 -v

# Example training script (saves checkpoint at ./checkpoint.pt).
python3 example_training.py

# Example inference script (needs checkpoint at ./checkpoint.pt).
echo "Good Job!" | python3 example_inference.py
```
