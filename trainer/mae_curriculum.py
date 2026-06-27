import lightning.pytorch as pl
import torch
import torchvision
from lightly.models import utils
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM
from lightly.transforms import MAETransform
from timm.models import VisionTransformer, create_model
from torch import Tensor, nn
from timm.scheduler import CosineLRScheduler
from torch.optim.lr_scheduler import LinearLR
from .mae import MAE
import math


class CurriculumMAE(MAE):
    def __init__(
        self,
        *args,
        tier_centers: dict[int, int] | None = None,
        annealing_sharpness: float = 8.0,
        weight_floor: float = 0.05,
        curriculum_masking: bool = True,
        **kwargs,
    ) -> None:
        self.save_hyperparameters()
        self.tier_centers = tier_centers
        self.annealing_sharpness = annealing_sharpness
        self.weight_floor = weight_floor
        self.curriculum_masking = curriculum_masking
        super().__init__(*args, **kwargs)

    def _current_mask_ratio(self) -> float:
        if not self.curriculum_masking:
            return self.mask_ratio
        t = self.current_epoch
        max_epochs = self.trainer.max_epochs or 50
        mask_ratio = 0.60 + 0.15 * (1 - math.cos(math.pi * t / max_epochs)) / 2
        self.log("mask_ratio", mask_ratio, on_step=False, on_epoch=True)
        return mask_ratio

    def _annealed_weight(self, tier: Tensor) -> Tensor:
        if self.tier_centers is None:
            return torch.ones_like(tier, dtype=torch.float32)
        t = self.current_epoch
        max_epochs = self.trainer.max_epochs or 50
        assert max_epochs > 0, "max_epochs must be set in the trainer"
        weights = torch.zeros_like(tier, dtype=torch.float32)
        for k, center in self.tier_centers.items():
            mask = tier == k
            if mask.any():
                w = torch.sigmoid(
                    tier.new_tensor(
                        self.annealing_sharpness * (t - center) / max_epochs,
                        dtype=torch.float32,
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

    def training_step(self, batch, batch_idx):
        self.mask_ratio = self._current_mask_ratio()
        weights = self._annealed_weight(batch["difficulty_tier"])

        with torch.no_grad():
            views = self.transform(batch["image"].float())
        images = views[0]  # views contains only a single view
        batch_size = images.shape[0]
        idx_keep, idx_mask = utils.random_token_mask(
            size=(batch_size, self.sequence_length),
            mask_ratio=self.mask_ratio,
            device=images.device,
        )
        x_encoded = self.forward_encoder(images=images, idx_keep=idx_keep)
        x_pred = self.forward_decoder(
            x_encoded=x_encoded, idx_keep=idx_keep, idx_mask=idx_mask
        )

        # get image patches for masked tokens
        patches = utils.patchify(images, self.patch_size)
        # must adjust idx_mask for missing class token
        target = utils.get_at_index(patches, idx_mask - 1)

        # per-sample reconstruction loss
        loss_per_sample = nn.functional.mse_loss(
            x_pred, target, reduction="none"
        )
        loss_per_sample = loss_per_sample.mean(
            dim=list(range(1, loss_per_sample.ndim))
        )  # (B,)

        weights = weights.to(
            loss_per_sample.device, dtype=loss_per_sample.dtype
        )
        weights_sum = weights.sum().clamp_min(1e-8)
        loss = (loss_per_sample * weights).sum() / weights_sum

        psnr = -10.0 * torch.log10(loss.detach().clamp(min=1e-10))

        self.log("train_loss", loss, on_step=True, on_epoch=True)
        # Near-zero std indicates that the model is learning uniformly across samples.
        self.log(
            "train_loss_std",
            loss_per_sample.std(),
            on_step=True,
            on_epoch=False,
        )
        # Near-zero means that the model is predicting a constant value, which is a common failure mode for MAE training.
        self.log("pred_std", x_pred.std(), on_step=True, on_epoch=False)
        # If this is very low, the model is getting "easy" patches (smooth background) and the loss won't be meaningful.
        self.log("target_std", target.std(), on_step=True, on_epoch=False)
        # PSNR is a common metric for image reconstruction quality. Higher is better, and values above ~30 indicate good reconstruction.
        self.log("psnr", psnr, on_step=False, on_epoch=True)

        return loss
