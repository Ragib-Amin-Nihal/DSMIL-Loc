import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class StableMILLoss(nn.Module):
    def __init__(self, 
                 pos_weight: torch.Tensor = None,
                 smooth_factor: float = 0.1,
                 focal_gamma: float = 2.0,
                 temporal_smoothness_weight: float = 0.1,
                 sparsity_weight: float = 0.01,
                 instance_consistency_weight: float = 0.1):
        super().__init__()
        # Main loss parameters
        self.pos_weight = torch.tensor([3.5]) if pos_weight is None else pos_weight
        self.smooth_factor = smooth_factor
        self.focal_gamma = focal_gamma
        
        # Additional loss weights
        self.temporal_smoothness_weight = temporal_smoothness_weight
        self.sparsity_weight = sparsity_weight
        self.instance_consistency_weight = instance_consistency_weight
    
    def forward(self, outputs: Dict[str, torch.Tensor], targets: torch.Tensor) -> torch.Tensor:
        logits = outputs['logits'].squeeze()
        targets = targets.float()
        
        # Label smoothing
        targets = targets * (1 - self.smooth_factor) + self.smooth_factor / 2
        
        # Focal loss with BCE
        pt = torch.exp(-F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        ))
        bce_loss = (1 - pt)**self.focal_gamma * F.binary_cross_entropy_with_logits(
            logits, targets,
            pos_weight=self.pos_weight,
            reduction='none'
        )
        
        total_loss = bce_loss.mean()
        
        # Add temporal smoothness loss if attention weights available
        if 'attention_weights' in outputs:
            attention_weights = outputs['attention_weights'].squeeze()
            temp_loss = self._temporal_smoothness_loss(attention_weights)
            sparsity_loss = self._compute_sparsity_loss(attention_weights)
            
            total_loss += (self.temporal_smoothness_weight * temp_loss + 
                          self.sparsity_weight * sparsity_loss)
        
        # Add instance consistency loss if embeddings available
        if 'instance_embeddings' in outputs:
            embeddings = outputs['instance_embeddings']
            instance_reg = self.instance_consistency_loss(embeddings)
            total_loss += self.instance_consistency_weight * instance_reg
        
        return total_loss
    
    def _temporal_smoothness_loss(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """Penalize sudden changes in attention weights"""
        # Calculate differences between adjacent weights
        diffs = attention_weights[:, 1:] - attention_weights[:, :-1]
        return torch.mean(diffs.pow(2))
    
    def _compute_sparsity_loss(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """Encourage selective attention"""
        return torch.mean(torch.abs(attention_weights))
    
    def instance_consistency_loss(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Encourage consistent embeddings within bags"""
        # Compute pairwise distances
        dist = torch.cdist(embeddings, embeddings)
        
        # Encourage closer distances for embeddings in same bag
        loss = torch.mean(dist)
        return loss
