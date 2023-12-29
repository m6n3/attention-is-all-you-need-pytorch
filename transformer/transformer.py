from transformer import encoder, decoder

import torch
import torch.nn as nn


class Transformer(nn.Module):
    def __init__(
        self,
        *,
        src_vocab,
        trg_vocab,
        hid_dim=512,  # embed_dim
        feedforward_dim=2048,
        num_enc_layers=6,
        num_dec_layers=6,
        num_attention_heads=8,
        max_seq_len=100,
        dropout=0.1,
        use_gpu=False,
    ):
        super().__init__()

        if use_gpu:
            assert (
                torch.cuda.is_available()
            ), "Error: no GPU device is available, consider setting `use_gpu` to False."
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu")

        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.encoder = encoder.Encoder(
            vocab_size=len(src_vocab),
            embed_dim=hid_dim,
            feedforward_dim=feedforward_dim,
            num_layers=num_enc_layers,
            num_heads=num_attention_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            device=self.device,
        )
        self.decoder = decoder.Decoder(
            vocab_size=len(trg_vocab),
            embed_dim=hid_dim,
            feedforward_dim=feedforward_dim,
            num_layers=num_dec_layers,
            num_heads=num_attention_heads,
            max_seq_len=max_seq_len,
            dropout=dropout,
            device=self.device,
        )

        # put model on right device.
        self.to(self.device)

    def get_device(self):
        return self.device

    def load_checkpoint(self, checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path))

    def save_checkpoint(self, checkpoint_path):
        torch.save(self.state_dict(), checkpoint_path)

    def make_src_mask(self, src):
        # src: [batch size, seq len]

        # This mask makes encoder to not pay attention to <PAD> tokens.
        #
        # Add two dimentions so it becomes compatible with tensors to which it
        # will be applied.
        # Hint: the tensor in MultiHeadedAttention module has shape
        # of [batch size, num heads, seq len, seq len].
        src_mask = (
            (src != self.src_vocab["<PAD>"]).unsqueeze(1).unsqueeze(2).to(self.device)
        )
        # pad_mask: [batch_size, 1, 1, seq len]

        return src_mask

    def make_trg_mask(self, trg):
        # trg: [batch size, seq len]

        # mask to ignore <PAD> token in trg sentence.
        pad_mask = (
            (trg != self.trg_vocab["<PAD>"]).unsqueeze(1).unsqueeze(2).to(self.device)
        )
        # pad_mask: [batch size, 1, 1, seq len]

        # mask to ignore future tokens (i.e., next words, hence a triangular shape).
        seq_len = trg.size(1)
        future_token_mask = (
            torch.tril(torch.ones(seq_len, seq_len)).bool().to(self.device)
        )
        # future_token_mask: [seq len, seq len]

        trg_mask = pad_mask & future_token_mask  # broadcasting magic.
        # trg_mask: [batch size, 1, seq len, seq len]

        return trg_mask

    def forward(self, src, trg):
        # src: [batch size, seq len]
        # trg: [batch size, seq len]

        src_mask, trg_mask = self.make_src_mask(src), self.make_trg_mask(trg)
        # src_mask = [batch size, 1, 1, seq len]
        # trg_mask = [batch size, 1, seq len, seq len]

        enc_out = self.encoder(src, src_mask)
        # enc_out: [batch size, seq len, hid dim]

        dec_out = self.decoder(trg, trg_mask, enc_out, src_mask)
        # dec_out: [batch size, seq len, hid dim]

        return dec_out
