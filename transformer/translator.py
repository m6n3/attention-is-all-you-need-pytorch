import torch


class Translator(object):
    def __init__(
        self, *, model, src_vocab, trg_vocab, src_tokenizer, max_trg_sentence_len
    ):
        self.model = model
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.src_tokenizer = src_tokenizer
        self.max_trg_sentence_len = max_trg_sentence_len
        self.device = model.get_device()

    def translate(self, sentence):
        src = (
            [self.src_vocab["<SOS>"]]
            + [self.src_vocab[tok] for tok in self.src_tokenizer(sentence)]
            + [self.src_vocab["<EOS>"]]
        )
        src_tensor = (
            torch.LongTensor(src).unsqueeze(0).to(self.device)
        )  # give batch dim to it.
        # src_tensor: [1, seq len]

        trg = [self.trg_vocab["<SOS>"]]

        for i in range(self.max_trg_sentence_len):
            trg_tensor = torch.LongTensor(trg).unsqueeze(0).to(self.device)

            pred = self.model(src_tensor, trg_tensor)
            # pred: [1, len(trg_tensor), trg_vocab size]

            last_word = pred.argmax(2)[0, -1].item()
            # last_word: scalar between 0 and trg_vocab size.

            trg.append(last_word)

            if last_word == self.trg_vocab["<EOS>"]:
                break

        itos = self.trg_vocab.get_itos()  # vocab int -> vocab word
        return " ".join([itos[w] for w in trg[1:-1]])
