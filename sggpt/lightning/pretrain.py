from dataclasses import dataclass
from functools import partial
from pathlib import Path

import datasets
import torch
from einops import rearrange, repeat
from pytorch_lightning import LightningModule
from torch import nn
from torch.utils.data import DataLoader

from ..models import *
from ..models.sggpt import init_weights_impl
from ..utils.collator import PretrainCollator
from ..utils.optim import get_cosine_schedule_with_warmup_and_min_lr_lambda


@dataclass
class PretrainConfig:
    dataset_path: str
    batch_size: int = 64
    global_batch_size: int = 1024
    lr: float = 1e-3
    min_lr: float = 1e-4
    adamw_betas: tuple[float, float] = (0.9, 0.99)
    adamw_eps: float = 1e-5
    weight_decay: float = 1e-2
    valid_ratio: float = 0.05
    n_epochs: int = 20
    num_warmup_steps: int = 200
    n_workers: int = 4
    seed: int = 42


class PretrainSelfGenomeGPT(LightningModule):
    def __init__(self, sggpt_config: dict, pretrain_config: dict):
        super().__init__()
        self.save_hyperparameters()
        self.sggpt_config = SelfGenomeGPTConfig(**sggpt_config)
        self.pretrain_config = PretrainConfig(**pretrain_config)

        self.model = SelfGenomeGPT(self.sggpt_config)
        init_fn = partial(
            init_weights_impl, initializer_range=self.sggpt_config.initializer_range
        )
        self.model.apply(init_fn)

        self.loss_fn = nn.CrossEntropyLoss()

        ds_path = Path(self.pretrain_config.dataset_path)
        if ds_path.exists():
            ds = datasets.load_from_disk(self.pretrain_config.dataset_path)
        else:
            ds = datasets.load_dataset(self.pretrain_config.dataset_path)
            if isinstance(ds, datasets.DatasetDict) and "train" in ds:
                ds = ds["train"]
            else:
                raise ValueError(
                    f"Invalid dataset path: {self.pretrain_config.dataset_path}"
                )
        split_ds = ds.train_test_split(
            test_size=self.pretrain_config.valid_ratio,
            shuffle=True,
            seed=self.pretrain_config.seed,
        )
        self.train_ds = split_ds["train"]
        self.val_ds = split_ds["test"]
        self.collator = PretrainCollator()
        self.loader_config = {
            "batch_size": self.pretrain_config.batch_size,
            "num_workers": self.pretrain_config.n_workers,
            "pin_memory": True,
            "drop_last": True,
            "collate_fn": self.collator.batch,
        }

    def forward(self, x: torch.LongTensor):
        out = self.model.forward(x)
        normed_logits = nn.functional.normalize(out.head_logits, dim=-1)
        foward_strand_logits, reverse_strand_logits = torch.chunk(
            normed_logits, chunks=2, dim=0
        )
        foward_strand_logits = rearrange(foward_strand_logits, "b l c -> l b c")
        reverse_strand_logits = rearrange(reverse_strand_logits, "b l c -> l c b")
        return foward_strand_logits, reverse_strand_logits

    def contrastive_loss(self, f: torch.Tensor, b: torch.Tensor):
        """
        f: (seq_len, batch, hidden)
        b: (seq_len, hidden, batch)
        """
        transformer_max_len = f.shape[0]  # e.g. 8
        batch_size = f.shape[1]

        ## if original seq is:
        #   ABCDEFGH
        ## seqlen = 8 and conv size = 2
        ## and the transformer_seqlen will be (8 / (2 / 2)) - 1 = 7

        ## then the forward stran on the seqlen dim will get the information of:
        #   0  1  2  3  4  5  6
        #   AB BC CD DE EF FG GH

        ## and the reverse strand will get the information of: (- means reverse complement)
        #   0   1   2   3   4   5   6
        #   -HG -GF -FE -ED -DC -CB -BA

        ## so the embedding we want to contrast is those not leaked by the conv size,
        ## but can compose to the original seq
        ## for example, if we take the forward[2, CD] element, the target (non overlapped)
        ## should be the reverse strand of forward[4, EF] (which is 2 + 2) element
        ## so the target should be reverse[2, -FE]

        ## based on the above derivation, we know the Fsrc-Rtarget index formula is:
        ## idx_reverse = (transformer_seqlen - 1) - (idx_forward + 2)
        ##             = (transformer_seqlen - 3) - idx_forward
        ##             = (7 - 3) - 2 = 2

        f = f[:-2]
        valid_l = f.shape[0]

        f_idx = torch.arange(valid_l, device=f.device)

        b_idx = (transformer_max_len - 3) - f_idx
        b = b[b_idx]  # re-index the reverse strand

        logits = torch.bmm(f, b)  # (valid_l, batch, batch)
        logits = rearrange(logits, "l a b -> (l a) b")

        target = repeat(
            torch.arange(batch_size, device=f.device), "b -> (l b)", l=valid_l
        )
        loss = self.loss_fn(
            logits,
            target,
        )
        acc = (logits.argmax(dim=-1) == target).float().mean()
        return loss, acc

    def training_step(self, batch, batch_idx):
        f, b = self(batch)
        loss, acc = self.contrastive_loss(f, b)
        self.log("train_loss", loss, on_step=True, on_epoch=True)
        self.log("train_acc", acc, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        f, b = self(batch)
        loss, acc = self.contrastive_loss(f, b)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            shuffle=True,
            **self.loader_config,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds,
            shuffle=False,
            **self.loader_config,
        )

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.pretrain_config.lr,
            betas=self.pretrain_config.adamw_betas,
            eps=self.pretrain_config.adamw_eps,
            weight_decay=self.pretrain_config.weight_decay,
        )
        training_steps = (
            len(self.train_ds)
            // self.pretrain_config.global_batch_size
            * self.pretrain_config.n_epochs
        )
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optim,
            lr_lambda=partial(
                get_cosine_schedule_with_warmup_and_min_lr_lambda,
                num_warmup_steps=self.pretrain_config.num_warmup_steps,
                num_training_steps=training_steps,
                num_cycles=0.5,
                lr=self.pretrain_config.lr,
                min_lr=self.pretrain_config.min_lr,
            ),
            last_epoch=-1,
        )
        return {
            "optimizer": optim,
            "lr_scheduler": {
                "scheduler": lr_scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }
