# Adapted from Lightly

import torch
import torch.nn.functional as F
from lightly.loss import NTXentLoss
from lightly.utils import dist
from torch import Tensor, nn


class NTXentLossWeighted(NTXentLoss):
    """NTXentLoss that optionally returns per-sample losses."""

    def __init__(
        self, temperature=0.5, memory_bank_size=0, gather_distributed=False
    ):
        super().__init__(temperature, memory_bank_size, gather_distributed)
        self.cross_entropy = nn.CrossEntropyLoss(reduction="none")

    def forward(
        self,
        out0: torch.Tensor,
        out1: torch.Tensor,
        sample_weights: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through Contrastive Cross-Entropy Loss.

        If used with a memory bank, the samples from the memory bank are used
        as negative examples. Otherwise, within-batch samples are used as
        negative samples.

        Args:
            out0:
                Output projections of the first set of transformed images.
                Shape: (batch_size, embedding_size)
            out1:
                Output projections of the second set of transformed images.
                Shape: (batch_size, embedding_size)

        Returns:
            Contrastive Cross Entropy Loss value.
        """

        device = out0.device
        batch_size, _ = out0.shape

        # Normalize the output to length 1
        out0 = nn.functional.normalize(out0, dim=1)
        out1 = nn.functional.normalize(out1, dim=1)

        # ask memory bank for negative samples and extend it with out1 if
        # out1 requires a gradient, otherwise keep the same vectors in the
        # memory bank (this allows for keeping the memory bank constant e.g.
        # for evaluating the loss on the test set)
        # out1: shape: (batch_size, embedding_size)
        # negatives: shape: (embedding_size, memory_bank_size)
        out1, negatives = self.memory_bank.forward(
            out1, update=out0.requires_grad
        )

        # Use cosine similarity (dot product) as all vectors are normalized to unit length
        # Notation in einsum: n = batch_size, c = embedding_size and k = memory_bank_size.

        if negatives is not None:
            # Use negatives from memory bank
            negatives = negatives.to(device)

            # sim_pos is of shape (batch_size, 1) and sim_pos[i] denotes the similarity
            # of the i-th sample in the batch to its positive pair
            sim_pos = torch.einsum("nc,nc->n", out0, out1).unsqueeze(-1)

            # sim_neg is of shape (batch_size, memory_bank_size) and sim_neg[i,j] denotes the similarity
            # of the i-th sample to the j-th negative sample
            sim_neg = torch.einsum("nc,ck->nk", out0, negatives)

            # Set the labels to maximize sim_pos in relation to sim_neg
            logits = torch.cat([sim_pos, sim_neg], dim=1) / self.temperature
            labels = torch.zeros(
                logits.shape[0], device=device, dtype=torch.long
            )

        else:
            # Use other samples from batch as negatives
            # and create diagonal mask that only selects similarities between
            # views of the same image
            if self.gather_distributed and dist.world_size() > 1:
                # Gather hidden representations from other processes
                out0_large = torch.cat(dist.gather(out0), 0)
                out1_large = torch.cat(dist.gather(out1), 0)
                diag_mask = dist.eye_rank(batch_size, device=out0.device)
            else:
                # Single process
                out0_large = out0
                out1_large = out1
                diag_mask = torch.eye(
                    batch_size, device=out0.device, dtype=torch.bool
                )

            # Calculate similiarities
            # Here n = batch_size and m = batch_size * world_size
            # The resulting vectors have shape (n, m)
            logits_00 = (
                torch.einsum("nc,mc->nm", out0, out0_large) / self.temperature
            )
            logits_01 = (
                torch.einsum("nc,mc->nm", out0, out1_large) / self.temperature
            )
            logits_10 = (
                torch.einsum("nc,mc->nm", out1, out0_large) / self.temperature
            )
            logits_11 = (
                torch.einsum("nc,mc->nm", out1, out1_large) / self.temperature
            )

            # Remove simliarities between same views of the same image
            logits_00 = logits_00[~diag_mask].view(batch_size, -1)
            logits_11 = logits_11[~diag_mask].view(batch_size, -1)

            # Concatenate logits
            # The logits tensor in the end has shape (2*n, 2*m-1)
            logits_0100 = torch.cat([logits_01, logits_00], dim=1)
            logits_1011 = torch.cat([logits_10, logits_11], dim=1)
            logits = torch.cat([logits_0100, logits_1011], dim=0)

            # Create labels
            labels = torch.arange(batch_size, device=device, dtype=torch.long)
            if self.gather_distributed:
                labels = labels + dist.rank() * batch_size
            labels = labels.repeat(2)

        # Calculate the cross-entropy loss
        loss = self.cross_entropy(logits, labels)
        if sample_weights is not None:
            sample_weights = sample_weights.to(loss.device)
            if (
                sample_weights.shape[0] == loss.shape[0] // 2
                and loss.shape[0] % 2 == 0
            ):
                sample_weights = sample_weights.repeat(2)

            # Normalize so weights sum to batch_size — prevents loss scale collapse
            sample_weights = sample_weights / (sample_weights.mean() + 1e-8)
            return (loss * sample_weights).mean()

        return loss.mean()