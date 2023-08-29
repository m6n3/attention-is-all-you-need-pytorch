import dataset as d

import torch
import torch.nn as nn
from torch.optim import Adam
from tqdm import tqdm


class Trainer(object):
    def __init__(
        self,
        *,
        model,
        dataset,
        train_batch_size=32,
        train_lr=8e-5,
        train_epochs=10,
        train_num_steps=1_000_000,
        save_every_n_steps=100,
        save_folder="./model",
    ):
        super().__init__()
        self.model = model
        self.dataloader = d.build_dataloader(
            dataset=dataset,
            src_vocab=dataset.get_src_vocab(),
            trg_vocab=dataset.get_trg_vocab(),
            batch_size=train_batch_size,
        )
        self.train_epochs = train_epochs
        self.train_num_steps = train_num_steps
        self.optim = Adam(self.model.parameters(), lr=train_lr)
        self.loss_fn = nn.CrossEntropyLoss(
            ignore_index=dataset.get_trg_vocab()["<PAD>"]
        )
        self.save_every_n_steps = save_every_n_steps
        self.save_folder = save_folder

    def train(self):
        self.model.train()
        best_loss = float("inf")
        num_steps = 0
        for epoch in range(self.train_epochs):
            running_loss = 0.0
            steps_in_running_loss = 0
            for idx, (src, trg) in enumerate(tqdm(self.dataloader)):
                num_steps += 1
                if num_steps > self.train_num_steps:
                    break

                self.optim.zero_grad()

                src, trg = src.permute(1, 0), trg.permute(1, 0)
                # src, trg: [batch size, seq len]

                # we do not supply trg <EOS> for training because in inference we want model predict <EOS> itself.
                # we ignore <SOS> for evaluating model prediction, because in inference we supply <SOS> to model, and expect prediction for subsequent words.
                trg_for_training = trg[:, :-1]
                trg_for_pred_eval = trg[:, 1:]
                pred = self.model(src, trg_for_training)

                # loss_fn expects input of shape [batch size * seq len, (trg) vocab size] and [batch size * seq len]
                loss = self.loss_fn(
                    pred.contiguous().view(-1, pred.size(2)),
                    trg_for_pred_eval.contiguous().view(-1),
                )
                loss.backward()
                # max norm of the gradients=1
                nn.utils.clip_grad_norm_(self.model.parameters(), 1)

                self.optim.step()

                running_loss += loss
                steps_in_running_loss += 1

                if (
                    self.save_folder
                    and (
                        idx % self.save_every_n_steps == 0
                        or num_steps == self.train_num_steps
                    )
                    and (running_loss / steps_in_running_loss) < best_loss
                ):
                    best_loss = running_loss / steps_in_running_loss
                    torch.save(self.model.state_dict(), self.save_folder)
                    running_loss = 0.0
                    steps_in_running_loss = 0
