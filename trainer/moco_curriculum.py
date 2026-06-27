# Adapted from TorchGeo

from typing import Any

import torch
import torch.nn.functional as F
from lightly.models.utils import update_momentum
from lightly.utils.scheduler import cosine_schedule
from torch import Tensor
from torchgeo.trainers import MoCoTask

from .losses import NTXentLossWeighted


class CurriculumMoCoTask(MoCoTask):
    def __init__(
        self,
        *args,
        tier_centers: dict[int, int] = {0: 0, 1: 10, 2: 20},
        annealing_sharpness: float = 8.0,
        weight_floor: float = 0.05,
        **kwargs,
    ) -> None:
        self.tier_centers = tier_centers
        self.annealing_sharpness = annealing_sharpness
        self.weight_floor = weight_floor
        super().__init__(*args, **kwargs)

    def configure_losses(self):
        self.criterion = NTXentLossWeighted(
            self.hparams["temperature"],
            (self.hparams["memory_bank_size"], self.hparams["output_dim"]),
            self.hparams["gather_distributed"],
        )

    def _annealed_weight(self, tier: Tensor) -> Tensor:
        t = self.current_epoch
        max_epochs = self.trainer.max_epochs or 50
        assert max_epochs > 0, "max_epochs must be set in the trainer"
        weights = torch.zeros_like(tier, dtype=torch.float32)
        for k, center in self.tier_centers.items():
            mask = tier == k
            if mask.any():
                w = torch.sigmoid(
                    torch.tensor(
                        self.annealing_sharpness * (t - center) / max_epochs
                    )
                ).clamp(min=self.weight_floor)
                weights[mask] = w
                self.log(
                    f"curriculum_weight/tier_{k}",
                    w.item(),
                    on_step=False,
                    on_epoch=True,
                    prog_bar=False,
                )
        return weights

    def training_step(
        self, batch: Any, batch_idx: int, dataloader_idx: int = 0
    ) -> Tensor:
        """Compute the training loss and additional metrics.

        Args:
            batch: The output of your DataLoader.
            batch_idx: Integer displaying index of this batch.
            dataloader_idx: Index of the current dataloader.

        Returns:
            The loss tensor.
        """
        weights = self._annealed_weight(batch["difficulty_tier"])

        x = batch["image"]
        batch_size = x.shape[0]

        in_channels = self.hparams["in_channels"]
        assert x.size(1) == in_channels or x.size(1) == 2 * in_channels

        if x.size(1) == in_channels:
            x1 = x
            x2 = x
        else:
            x1 = x[:, :in_channels]
            x2 = x[:, in_channels:]

        with torch.no_grad():
            x1 = self.augmentation1(x1)
            x2 = self.augmentation2(x2)

        m = self.hparams["moco_momentum"]
        if self.hparams["version"] == 1:
            q, h1 = self.forward(x1)
            with torch.no_grad():
                update_momentum(self.backbone, self.backbone_momentum, m)
                k = self.forward_momentum(x2)
            loss: Tensor = self.criterion(q, k, weights)
        elif self.hparams["version"] == 2:
            q, h1 = self.forward(x1)
            with torch.no_grad():
                update_momentum(self.backbone, self.backbone_momentum, m)
                update_momentum(
                    self.projection_head, self.projection_head_momentum, m
                )
                k = self.forward_momentum(x2)
            loss = self.criterion(q, k, weights)
        if self.hparams["version"] == 3:
            max_steps = self.trainer.max_epochs or 200
            m = cosine_schedule(self.current_epoch, max_steps, m, 1)
            q1, h1 = self.forward(x1)
            q2, _ = self.forward(x2)
            with torch.no_grad():
                update_momentum(self.backbone, self.backbone_momentum, m)
                update_momentum(
                    self.projection_head, self.projection_head_momentum, m
                )
                k1 = self.forward_momentum(x1)
                k2 = self.forward_momentum(x2)
            loss = self.criterion(q1, k2, weights) + self.criterion(
                q2, k1, weights
            )

        # Calculate the mean normalized standard deviation over features dimensions.
        # If this is << 1 / sqrt(h1.shape[1]), then the model is not learning anything.
        output = h1.detach()
        output = F.normalize(output, dim=1)
        output_std = torch.std(output, dim=0)
        output_std = torch.mean(output_std, dim=0)
        self.avg_output_std = (
            0.9 * self.avg_output_std + (1 - 0.9) * output_std.item()
        )

        self.log("train_ssl_std", self.avg_output_std, batch_size=batch_size)
        self.log("train_loss", loss, batch_size=batch_size)

        return loss
