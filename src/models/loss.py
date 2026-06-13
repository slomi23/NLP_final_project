"""Loss functions for neural search / contrastive learning."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
    """Classic pairwise contrastive loss.

    labels: 1 for similar pairs, 0 for dissimilar pairs.
    """

    def __init__(self, margin: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        labels = labels.float()
        distances = F.pairwise_distance(embeddings1, embeddings2, p=2)

        loss_similar = labels * distances.pow(2)
        loss_dissimilar = (1.0 - labels) * F.relu(self.margin - distances).pow(2)
        loss = 0.5 * (loss_similar + loss_dissimilar)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class NTXentLoss(nn.Module):
    """SimCLR NT-Xent loss.

    projections shape: [2 * batch_size, embed_dim]
    First half and second half are positive views of each other.
    """

    def __init__(self, temperature: float = 0.07, reduction: str = "mean"):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, projections: torch.Tensor) -> torch.Tensor:
        n = projections.shape[0]
        if n % 2 != 0:
            raise ValueError("NTXentLoss expects an even number of projections: [2 * batch_size, dim]")

        batch_size = n // 2
        projections = F.normalize(projections, dim=1)

        logits = torch.matmul(projections, projections.T) / self.temperature

        # Remove self-comparisons from denominator.
        self_mask = torch.eye(n, dtype=torch.bool, device=projections.device)
        logits = logits.masked_fill(self_mask, torch.finfo(logits.dtype).min)

        targets = torch.cat(
            [
                torch.arange(batch_size, 2 * batch_size, device=projections.device),
                torch.arange(0, batch_size, device=projections.device),
            ]
        )

        loss = F.cross_entropy(logits, targets, reduction=self.reduction)
        return loss


class InfoNCELoss(nn.Module):
    """InfoNCE loss for query-positive-negatives training.

    anchor:    [batch_size, embed_dim]
    positive:  [batch_size, embed_dim]
    negatives: [batch_size, num_negatives, embed_dim]
    """

    def __init__(self, temperature: float = 0.07, reduction: str = "mean"):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        if negatives.dim() != 3:
            raise ValueError("negatives must have shape [batch_size, num_negatives, embed_dim]")

        batch_size = anchor.shape[0]
        if positive.shape[0] != batch_size or negatives.shape[0] != batch_size:
            raise ValueError("anchor, positive, and negatives must have the same batch size")

        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        negatives = F.normalize(negatives, dim=2)

        pos_logits = torch.sum(anchor * positive, dim=1, keepdim=True)
        neg_logits = torch.sum(anchor.unsqueeze(1) * negatives, dim=2)
        logits = torch.cat([pos_logits, neg_logits], dim=1) / self.temperature

        targets = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)
        return F.cross_entropy(logits, targets, reduction=self.reduction)


class SupervisedContrastiveLoss(nn.Module):
    """Supervised contrastive loss.

    features: [batch_size, embed_dim]
    labels:   [batch_size]
    """

    def __init__(self, temperature: float = 0.07, reduction: str = "mean"):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction

    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        features = F.normalize(features, dim=1)
        labels = labels.view(-1)
        batch_size = features.shape[0]

        logits = torch.matmul(features, features.T) / self.temperature

        self_mask = torch.eye(batch_size, dtype=torch.bool, device=features.device)
        positive_mask = labels.unsqueeze(0).eq(labels.unsqueeze(1)) & ~self_mask

        logits = logits - logits.max(dim=1, keepdim=True).values.detach()
        exp_logits = torch.exp(logits) * (~self_mask).float()
        log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True) + 1e-12)

        positives_per_row = positive_mask.sum(dim=1)
        valid_rows = positives_per_row > 0

        if not valid_rows.any():
            return torch.tensor(0.0, dtype=features.dtype, device=features.device, requires_grad=True)

        loss = -(positive_mask.float() * log_prob).sum(dim=1) / positives_per_row.clamp(min=1)
        loss = loss[valid_rows]

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


class LiftedStructureLoss(nn.Module):
    """Simple lifted-structure style loss.

    This is included for completeness, but your notebook should use InfoNCELoss.
    """

    def __init__(self, margin: float = 1.0, reduction: str = "mean"):
        super().__init__()
        self.margin = margin
        self.reduction = reduction

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        embeddings = F.normalize(embeddings, dim=1)
        labels = labels.view(-1)
        batch_size = embeddings.shape[0]

        distances = torch.cdist(embeddings, embeddings, p=2)
        same = labels.unsqueeze(0).eq(labels.unsqueeze(1))
        eye = torch.eye(batch_size, dtype=torch.bool, device=embeddings.device)
        positive_mask = same & ~eye
        negative_mask = ~same

        losses = []
        for i in range(batch_size):
            pos_distances = distances[i][positive_mask[i]]
            neg_distances = distances[i][negative_mask[i]]

            if pos_distances.numel() == 0 or neg_distances.numel() == 0:
                continue

            hardest_negative_term = torch.logsumexp(self.margin - neg_distances, dim=0)
            losses.append(F.relu(pos_distances + hardest_negative_term).pow(2).mean())

        if not losses:
            return torch.tensor(0.0, dtype=embeddings.dtype, device=embeddings.device, requires_grad=True)

        loss = torch.stack(losses)
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def create_loss(loss_name: str, **kwargs) -> nn.Module:
    """Factory function for contrastive losses."""
    name = loss_name.lower()

    if name in {"contrastive", "contrastive_loss"}:
        kwargs.pop("temperature", None)
        return ContrastiveLoss(**kwargs)
    if name in {"nt_xent", "ntxent", "simclr"}:
        return NTXentLoss(**kwargs)
    if name in {"info_nce", "infonce", "info_nce_loss"}:
        return InfoNCELoss(**kwargs)
    if name in {"supervised_contrastive", "supcon"}:
        return SupervisedContrastiveLoss(**kwargs)
    if name in {"lifted_structure", "lifted"}:
        kwargs.pop("temperature", None)
        return LiftedStructureLoss(**kwargs)

    raise ValueError(f"Unknown loss function: {loss_name}")
