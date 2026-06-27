import os

import comet_ml
import hydra
import kornia.augmentation as K
import pytorch_lightning as pl
import torch
from datasets import HDF5SSL4EODM
from kornia.augmentation import AugmentationSequential, Normalize
from lightly.transforms import MAETransform
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    Callback,
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import CometLogger
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from ssl4eos12_dataset import statistics
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchgeo.trainers import MoCoTask

from trainer import MAE, CurriculumMAE

CHECKPOINTS_FOLDER = ""


@hydra.main(
    config_path="configs", config_name="mae_vit_curriculum", version_base=None
)
def main(config: OmegaConf) -> None:
    torch.set_float32_matmul_precision("medium")
    seed_everything(42)
    # Initialize the data module for SSL4EOS12 dataset
    datamodule = HDF5SSL4EODM(**config.dataset, transforms=None)

    # Initialize the MAE pretraining task
    mae_transform = MAETransform(
        normalize={
            "mean": torch.tensor(statistics["mean"]["S2L1C"]),
            "std": torch.tensor(statistics["std"]["S2L1C"]),
        }
    )
    task = CurriculumMAE(**config.mae, transform=mae_transform)

    logger = CometLogger(project_name="CL-SSL")
    os.makedirs(
        f"{CHECKPOINTS_FOLDER}/{logger.experiment.get_key()}", exist_ok=True
    )
    checkpointer = ModelCheckpoint(
        f"{CHECKPOINTS_FOLDER}/{logger.experiment.get_key()}",
        every_n_epochs=20,
        save_top_k=-1,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor()

    # Train the model using the data module
    trainer = Trainer(
        **config.trainer,
        logger=logger,
        callbacks=[checkpointer, lr_monitor],
    )
    trainer.fit(task, datamodule=datamodule)


if __name__ == "__main__":
    main()
