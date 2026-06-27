import os

import comet_ml
import hydra
import kornia.augmentation as K
import torch
from datasets import HDF5SSL4EODM
from kornia.augmentation import AugmentationSequential, Normalize
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import Callback, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import CometLogger
from omegaconf import OmegaConf
from pytorch_lightning import seed_everything
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchgeo.trainers import MoCoTask


class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return torch.squeeze(x, dim=self.dim)


@hydra.main(config_path="configs", config_name="mocov2", version_base=None)
def main(config: OmegaConf) -> None:
    torch.set_float32_matmul_precision("medium")
    seed_everything(42)
    # Initialize the data module for SSL4EOS12 dataset
    datamodule = HDF5SSL4EODM(**config.dataset)

    # Initialize the MoCo pretraining task
    task = MoCoTask(**config.moco)

    logger = CometLogger(project_name="CL-SSL")
    os.makedirs(f"models/{logger.experiment.get_key()}", exist_ok=True)
    checkpointer = ModelCheckpoint(
        f"models/{logger.experiment.get_key()}",
        every_n_train_steps=246144
        // config.dataset.batch_size
        * 5,  # Save every 5 epochs
        save_top_k=-1,
        save_last=True,
    )
    lr_monitor = LearningRateMonitor()

    # Train the model using the data module
    trainer = Trainer(
        **config.trainer, logger=logger, callbacks=[checkpointer, lr_monitor]
    )
    trainer.fit(task, datamodule=datamodule)


if __name__ == "__main__":
    main()
