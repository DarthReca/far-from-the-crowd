# Adapted from Lightly

import lightning.pytorch as pl
import torch
import torchvision
from lightly.models import utils
from lightly.models.modules import MAEDecoderTIMM, MaskedVisionTransformerTIMM
from lightly.transforms import MAETransform
from timm.models import VisionTransformer, create_model
from timm.scheduler import CosineLRScheduler
from torch import nn
from torch.optim.lr_scheduler import LinearLR


class MAE(pl.LightningModule):
    def __init__(
        self,
        vit_name: str = "vit_base_patch32_224",
        transform: nn.Module | None = None,
        decoder_dim: int = 512,
        lr: float = 1.5e-4,
        decoder_num_heads: int = 8,
        weight_decay: float = 0.05,
    ) -> None:
        """MAE training

        Args:
            lr (float, optional): _Should be 1.5e-4 * batch_size / 256.
        """
        super().__init__()
        self.transform = transform if transform is not None else MAETransform()

        vit = create_model(vit_name, pretrained=False, in_chans=13)
        assert isinstance(vit, VisionTransformer), (
            "Only ViT models are supported"
        )
        self.mask_ratio = 0.75
        self.patch_size = vit.patch_embed.patch_size[0]
        self.backbone = MaskedVisionTransformerTIMM(vit=vit)
        self.sequence_length = self.backbone.sequence_length
        self.decoder = MAEDecoderTIMM(
            num_patches=vit.patch_embed.num_patches,
            patch_size=self.patch_size,
            embed_dim=vit.embed_dim,
            decoder_embed_dim=decoder_dim,
            decoder_depth=1,
            decoder_num_heads=decoder_num_heads,
            mlp_ratio=4.0,
            proj_drop_rate=0.0,
            attn_drop_rate=0.0,
            in_chans=vit.patch_embed.proj.in_channels,
        )
        self.criterion = nn.MSELoss()

        self.lr = lr
        self.weight_decay = weight_decay

    def forward_encoder(self, images, idx_keep=None):
        return self.backbone.encode(images=images, idx_keep=idx_keep)

    def forward_decoder(self, x_encoded, idx_keep, idx_mask):
        # build decoder input
        batch_size = x_encoded.shape[0]
        x_decode = self.decoder.embed(x_encoded)
        x_masked = utils.repeat_token(
            self.decoder.mask_token, (batch_size, self.sequence_length)
        )
        x_masked = utils.set_at_index(
            x_masked, idx_keep, x_decode.type_as(x_masked)
        )

        # decoder forward pass
        x_decoded = self.decoder.decode(x_masked)

        # predict pixel values for masked tokens
        x_pred = utils.get_at_index(x_decoded, idx_mask)
        x_pred = self.decoder.predict(x_pred)
        return x_pred

    def training_step(self, batch, batch_idx):
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

        loss = self.criterion(x_pred, target)

        # per-sample loss for std logging
        loss_per_sample = nn.functional.mse_loss(
            x_pred, target, reduction="none"
        )
        loss_per_sample = loss_per_sample.mean(
            dim=list(range(1, loss_per_sample.ndim))
        )  # (B,)
        loss = loss_per_sample.mean()

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

    def configure_optimizers(self):
        optim = torch.optim.AdamW(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = LinearLR(optim, start_factor=1e-4, total_iters=3)
        return {
            "optimizer": optim,
            "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
        }
