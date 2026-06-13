# loss.py (simple working version)
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """
    Simple working Contrastive Loss based on TensorFlow Addons implementation
    Formula: L = 0.5 * Y * D^2 + 0.5 * (Y-1) * {max(0, margin - D)}^2
    """
    
    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, embeddings1: torch.Tensor, embeddings2: torch.Tensor, 
                labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings1: [batch_size, embed_dim] - requires_grad=True
            embeddings2: [batch_size, embed_dim] - requires_grad=True  
            labels: [batch_size] - 1 for similar, 0 for dissimilar
        """
        # Calculate euclidean distance
        diff = embeddings1 - embeddings2
        distances = torch.sqrt(torch.sum(diff**2, dim=1) + 1e-8)  # Add epsilon for numerical stability
        
        # Contrastive loss formula
        loss_similar = labels * distances**2
        loss_dissimilar = (1 - labels) * torch.pow(F.relu(self.margin - distances), 2)
        
        loss = loss_similar + loss_dissimilar
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class NTXentLoss(nn.Module):
    """
    Simple working NT-Xent Loss based on SimCLR
    """
    
    def __init__(self, temperature: float = 0.07, reduction: str = 'mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(self, projections: torch.Tensor) -> torch.Tensor:
        """
        Args:
            projections: [2*batch_size, embed_dim] - requires_grad=True
                         (first half: view 1, second half: view 2)
        """
        batch_size = projections.shape[0] // 2
        
        # Normalize projections
        projections = F.normalize(projections, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(projections, projections.t()) / self.temperature
        
        # Create positive pairs mask
        pos_mask = torch.zeros(projections.shape[0], projections.shape[0], 
                              device=projections.device, dtype=torch.bool)
        for i in range(batch_size):
            pos_mask[i, i + batch_size] = True
            pos_mask[i + batch_size, i] = True
        
        # Remove self-similarity
        identity_mask = torch.eye(projections.shape[0], dtype=torch.bool, 
                                 device=projections.device)
        pos_mask = pos_mask & ~identity_mask
        
        # Get positive similarities
        pos_sim = similarity_matrix[pos_mask]
        
        # Get all similarities for denominator
        all_sim = similarity_matrix[~identity_mask]
        
        # InfoNCE loss
        exp_all = torch.exp(all_sim)
        sum_exp = exp_all.sum()
        
        loss = -torch.log(torch.exp(pos_sim).sum() / (sum_exp + 1e-8) + 1e-8)
        
        if self.reduction == 'mean':
            return loss
        elif self.reduction == 'sum':
            return loss
        else:
            return loss

class InfoNCELoss(nn.Module):
    """
    Simple working InfoNCE Loss
    """
    
    def __init__(self, temperature: float = 0.07, reduction: str = 'mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(self, anchor: torch.Tensor, positive: torch.Tensor, 
                negatives: torch.Tensor) -> torch.Tensor:
        """
        Args:
            anchor: [batch_size, embed_dim] - requires_grad=True
            positive: [batch_size, embed_dim] - requires_grad=True
            negatives: [batch_size, num_negatives, embed_dim] - requires_grad=True
        """
        batch_size, num_negatives = negatives.shape[0], negatives.shape[1]
        
        # Normalize embeddings
        anchor = F.normalize(anchor, dim=1)
        positive = F.normalize(positive, dim=1)
        negatives = F.normalize(negatives, dim=2)
        
        # Calculate similarities
        pos_sim = torch.sum(anchor * positive, dim=1) / self.temperature  # [batch_size]
        neg_sim = torch.sum(negatives * anchor.unsqueeze(1), dim=2) / self.temperature  # [batch_size, num_negatives]
        
        # Combine similarities
        all_sim = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)  # [batch_size, 1 + num_negatives]
        
        # InfoNCE loss
        return F.cross_entropy(all_sim, torch.zeros(batch_size, dtype=torch.long, device=anchor.device))

class SupervisedContrastiveLoss(nn.Module):
    """
    Simple working Supervised Contrastive Loss
    """
    
    def __init__(self, temperature: float = 0.07, reduction: str = 'mean'):
        super().__init__()
        self.temperature = temperature
        self.reduction = reduction
    
    def forward(self, features: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: [batch_size, embed_dim] - requires_grad=True
            labels: [batch_size] - class labels
        """
        batch_size = features.shape[0]
        
        # Normalize features
        features = F.normalize(features, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.mm(features, features.t()) / self.temperature
        
        # Create mask for positive pairs (same class)
        pos_mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
        pos_mask = pos_mask - torch.eye(batch_size, dtype=torch.float, device=features.device)
        
        # Compute loss
        pos_sim = similarity_matrix * pos_mask
        mean_pos_sim = (pos_sim.sum(1) / (pos_mask.sum(1) + 1e-8))
        
        exp_sim = torch.exp(similarity_matrix)
        
        loss = -torch.log(torch.exp(mean_pos_sim) / (exp_sim.sum(1) + 1e-8))
        
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

class LiftedStructureLoss(nn.Module):
    """
    Simple working Lifted Structure Loss
    """
    
    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction
    
    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            embeddings: [batch_size, embed_dim] - requires_grad=True
            labels: [batch_size] - 1 for similar, 0 for dissimilar
        """
        batch_size = embeddings.shape[0]
        
        # Normalize embeddings
        embeddings = F.normalize(embeddings, dim=1)
        
        # Compute pairwise similarities
        similarities = torch.mm(embeddings, embeddings.t())
        
        # Create mask for positive pairs
        pos_mask = (labels.unsqueeze(1) == labels.unsqueeze(0)).float()
        pos_mask = pos_mask - torch.eye(batch_size, dtype=torch.float, device=embeddings.device)
        
        loss = 0.0
        valid_pairs = 0
        
        for i in range(batch_size):
            if pos_mask[i].sum() == 0:
                continue
                
            # Positive similarities for sample i
            pos_sim = similarities[i][pos_mask[i] > 0]
            
            # Negative similarities for sample i
            neg_sim = similarities[i][pos_mask[i] == 0]
            
            # Compute hinge loss
            if len(pos_sim) > 0 and len(neg_sim) > 0:
                hardest_neg = neg_sim.max()
                loss += torch.sum(torch.log(1 + torch.exp(pos_sim - hardest_neg)))
                valid_pairs += len(pos_sim)
        
        if valid_pairs > 0:
            loss = loss / valid_pairs
        else:
            loss = torch.tensor(0.0, device=embeddings.device)
        
        return loss

# Factory function
def create_loss(loss_name: str, **kwargs) -> nn.Module:
    """Factory function to create contrastive loss"""
    loss_name = loss_name.lower()
    
    if loss_name == 'contrastive':
        return ContrastiveLoss(**{k: v for k, v in kwargs.items() if k != 'temperature'})
    elif loss_name == 'nt_xent':
        return NTXentLoss(**kwargs)
    elif loss_name == 'info_nce':
        return InfoNCELoss(**kwargs)
    elif loss_name == 'supervised_contrastive':
        return SupervisedContrastiveLoss(**kwargs)
    elif loss_name == 'lifted_structure':
        return LiftedStructureLoss(**{k: v for k, v in kwargs.items() if k != 'temperature'})
    else:
        raise ValueError(f"Unknown loss function: {loss_name}")

# Aliases for backward compatibility
ContrastiveLoss = ContrastiveLoss
NTXentLoss = NTXentLoss
InfoNCELoss = InfoNCELoss
SupervisedContrastiveLoss = SupervisedContrastiveLoss
LiftedStructureLoss = LiftedStructureLoss
