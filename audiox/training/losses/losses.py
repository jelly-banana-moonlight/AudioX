import typing as tp

from torch.nn import functional as F
from torch import nn
import torch
class LossModule(nn.Module):
    def __init__(self, name: str, weight: float = 1.0):
        super().__init__()

        self.name = name
        self.weight = weight

    def forward(self, info, *args, **kwargs):
        raise NotImplementedError
    
class ValueLoss(LossModule):
    def __init__(self, key: str, name, weight: float = 1.0):
        super().__init__(name=name, weight=weight)

        self.key = key
    
    def forward(self, info):
        return self.weight * info[self.key]

class L1Loss(LossModule):
    def __init__(self, key_a: str, key_b: str, weight: float = 1.0, mask_key: str = None, name: str = 'l1_loss'):
        super().__init__(name=name, weight=weight)

        self.key_a = key_a
        self.key_b = key_b

        self.mask_key = mask_key
    
    def forward(self, info):
        mse_loss = F.l1_loss(info[self.key_a], info[self.key_b], reduction='none')    

        if self.mask_key is not None and self.mask_key in info:
            mse_loss = mse_loss[info[self.mask_key]]

        mse_loss = mse_loss.mean()

        return self.weight * mse_loss
    
class MSELoss(LossModule):
    def __init__(self, key_a: str, key_b: str, weight: float = 1.0, mask_key: str = None, name: str = 'mse_loss'):
        super().__init__(name=name, weight=weight)

        self.key_a = key_a
        self.key_b = key_b

        self.mask_key = mask_key
    
    def forward(self, info):
        mse_loss = F.mse_loss(info[self.key_a], info[self.key_b], reduction='none')    

        if self.mask_key is not None and self.mask_key in info and info[self.mask_key] is not None:
            mask = info[self.mask_key]

            if mask.ndim == 2 and mse_loss.ndim == 3:
                mask = mask.unsqueeze(1)

            if mask.shape[1] != mse_loss.shape[1]:
                mask = mask.repeat(1, mse_loss.shape[1], 1)

            mse_loss = mse_loss[mask]

        mse_loss = mse_loss.mean()

        return self.weight * mse_loss
    
class MSELoss_align_weight(LossModule):
    def __init__(self, key_a: str, key_b: str, weight: float = 1.0, mask_key: str = None, name: str = 'mse_loss'):
        super().__init__(name=name, weight=weight)

        self.key_a = key_a
        self.key_b = key_b

        self.mask_key = mask_key
    
    def forward(self, info):
        mse_loss = F.mse_loss(info[self.key_a], info[self.key_b], reduction='none')    
        alpha = 0.2
        
        align_weight = info['av-align_weight'].unsqueeze(1).repeat(1, mse_loss.shape[1], 1)

        scaling_mask = torch.ones_like(mse_loss)

        scaling_mask[:, :, :215] = align_weight

        mse_loss_align_weight = mse_loss * torch.where(scaling_mask == 0, alpha, scaling_mask)

        if self.mask_key is not None and self.mask_key in info and info[self.mask_key] is not None:
            mask = info[self.mask_key]

            if mask.ndim == 2 and mse_loss_align_weight.ndim == 3:
                mask = mask.unsqueeze(1)

            if mask.shape[1] != mse_loss_align_weight.shape[1]:
                mask = mask.repeat(1, mse_loss_align_weight.shape[1], 1)

            mse_loss_align_weight = mse_loss_align_weight[mask]

        mse_loss_align_weight = mse_loss_align_weight.mean()

        return self.weight * mse_loss_align_weight
    
class CosineEmbeddingLoss(LossModule):
    def __init__(self, input_key: str, target_key: str, weight: float = 1.0, margin: float = 0.0, reduction: str = "mean", name: str = 'cosine_embedding_loss'):
        super().__init__(name=name, weight=weight)

        self.input_key = input_key
        self.target_key = target_key
        self.margin = margin
        self.reduction = reduction
        
        self.loss_fn = nn.CosineEmbeddingLoss(margin=margin, reduction=reduction)
    
    def forward(self, info):
        input_tensor = info[self.input_key]
        target_tensor = info[self.target_key]
        
        target_labels = torch.ones(input_tensor.shape[0], device=input_tensor.device)
        
        if input_tensor.ndim > 2:
            input_tensor = input_tensor.reshape(input_tensor.shape[0], -1)
        if target_tensor.ndim > 2:
            target_tensor = target_tensor.reshape(target_tensor.shape[0], -1)

        # min_feature_dim = min(input_tensor.shape[1], target_tensor.shape[1])
        # input_tensor = input_tensor[:, :min_feature_dim]
        # target_tensor = target_tensor[:, :min_feature_dim]
        
        input_tensor = F.normalize(input_tensor, p=2, dim=-1)
        target_tensor = F.normalize(target_tensor, p=2, dim=-1)
        
        loss = self.loss_fn(input_tensor, target_tensor, target_labels)
        
        return self.weight * loss

class AuralossLoss(LossModule):
    def __init__(self, auraloss_module, input_key: str, target_key: str, name: str, weight: float = 1):
        super().__init__(name, weight)

        self.auraloss_module = auraloss_module

        self.input_key = input_key
        self.target_key = target_key

    def forward(self, info):
        loss = self.auraloss_module(info[self.input_key], info[self.target_key])

        return self.weight * loss
    
class MultiLoss(nn.Module):
    def __init__(self, losses: tp.List[LossModule]):
        super().__init__()

        self.losses = nn.ModuleList(losses)

    def forward(self, info):
        total_loss = 0

        losses = {}

        for loss_module in self.losses:
            module_loss = loss_module(info)
            total_loss += module_loss
            losses[loss_module.name] = module_loss

        return total_loss, losses