from pathlib import Path

import pytorch_lightning as pl
import torch
import typer
import wandb
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichModelSummary,
    RichProgressBar,
)
from pytorch_lightning.loggers import CSVLogger, WandbLogger

from .lightning.pretrain import (
    PretrainConfig,
    PretrainSelfGenomeGPT,
    SelfGenomeGPTConfig,
)
from .utils.helpers import get_time_str

app = typer.Typer(pretty_exceptions_enable=False)


@app.command()
def train(
    # pretrain config
    dataset_path: str,
    batch_size: int = 64,
    global_batch_size: int = 1024,
    lr: float = 1e-3,
    min_lr: float = 1e-4,
    adamw_betas: tuple[float, float] = (0.9, 0.99),
    adamw_eps: float = 1e-5,
    weight_decay: float = 1e-2,
    valid_ratio: float = 0.05,
    n_epochs: int = 20,
    num_warmup_steps: int = 200,
    n_workers: int = 4,
    seed: int = 42,
    # model config
    conv_size: int = 40,
    d_model: int = 256,
    n_heads: int = 4,
    intermediate_size: int = 1024,
    n_layers: int = 4,
    norm_eps: float = 1e-12,
    max_len: int = 1000,
    initializer_range: float = 0.02,
    attention_dropout: float = 0.0,
    mlp_dropout: float = 0.0,
    attn_impl: str = "flash",
    # logger config
    logdir: Path = Path("./data/mammoth/logs/pretrain"),
    log_step: int = 20,
    project_name: str = "selfgenomegpt",
    # trainer config
    devices: int = 1,
    accelerator: str = "gpu",
    precision: str = "bf16-mixed",
    gradient_clip_val: float = 1.0,
    dev: bool = False,
    resume_ckpt: Path = None,
):
    assert (
        global_batch_size % (batch_size * devices) == 0
    ), "global_batch_size must be divisible by (batch_size * devices)"
    accumulator = global_batch_size // (batch_size * devices)
    pl.seed_everything(seed)

    version = get_time_str()
    logdir.mkdir(exist_ok=True, parents=True)
    wandb.login()

    print("version:", version)
    pretrain_config = PretrainConfig(
        dataset_path=dataset_path,
        batch_size=batch_size,
        global_batch_size=global_batch_size,
        lr=lr,
        min_lr=min_lr,
        adamw_betas=adamw_betas,
        adamw_eps=adamw_eps,
        weight_decay=weight_decay,
        valid_ratio=valid_ratio,
        n_epochs=n_epochs,
        num_warmup_steps=num_warmup_steps,
        n_workers=n_workers,
        seed=seed,
    )
    sggpt_config = SelfGenomeGPTConfig(
        conv_size=conv_size,
        d_model=d_model,
        n_heads=n_heads,
        intermediate_size=intermediate_size,
        n_layers=n_layers,
        norm_eps=norm_eps,
        max_len=max_len,
        initializer_range=initializer_range,
        attention_dropout=attention_dropout,
        mlp_dropout=mlp_dropout,
        attn_impl=attn_impl,
    )
    module = PretrainSelfGenomeGPT(
        sggpt_config=sggpt_config.__dict__, pretrain_config=pretrain_config.__dict__
    )

    print("-- setup loggers --")
    wandb_logger = WandbLogger(
        name=f"{project_name}-pretrain-{version}",
        save_dir=logdir,
        project=f"{project_name}",
    )
    csv_logger = CSVLogger(
        save_dir=logdir,
        name=f"{project_name}_csv_logs",
        version=version,
    )
    logger_dir = Path(csv_logger.log_dir)

    # callbacks
    lr_monitor = LearningRateMonitor(logging_interval="step")
    model_checkpoint = ModelCheckpoint(
        monitor="val_loss",
        dirpath=logger_dir / "epoch_ckpts",
        filename=project_name + "-pretrain-{epoch:02d}-{val_loss:.5f}",
        save_top_k=-1,
        save_last="link",
        mode="min",
    )
    rich_progress_bar = RichProgressBar()
    rich_model_summary = RichModelSummary(max_depth=2)
    callbacks = [
        model_checkpoint,
        lr_monitor,
        rich_progress_bar,
        rich_model_summary,
    ]

    print("-- setup trainer --")
    trainer = pl.Trainer(
        devices=devices,
        accelerator=accelerator,
        precision=precision,
        gradient_clip_val=gradient_clip_val,
        accumulate_grad_batches=accumulator,
        max_epochs=n_epochs,
        callbacks=callbacks,
        logger=[wandb_logger, csv_logger],
        log_every_n_steps=log_step,
        fast_dev_run=dev,
    )
    trainer.fit(module, ckpt_path=resume_ckpt)


@app.command()
def to_pytorch(
    ckpt_file: Path,
    output_model_dir: Path = None,
):
    pytorch_ckpts = ckpt_file.parent.parent / "pytorch_ckpts"
    output_model_dir = (
        pytorch_ckpts / ckpt_file.stem if output_model_dir is None else output_model_dir
    )
    model = PretrainSelfGenomeGPT.load_from_checkpoint(ckpt_file, map_location="cpu")
    model.model.save_pretrained(output_model_dir)


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")
    app()
