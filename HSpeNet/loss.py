import numpy as np
import torch
import torch.nn as nn

class SAMLoss(nn.Module):
    def __init__(self, reduction='mean'):
        super(SAMLoss, self).__init__()
        self.reduction = reduction
        self.pi = np.pi
        self.eps = 1e-8

    def forward(self, outputs, labels):

        norm_outputs = torch.sum(outputs * outputs, dim=1)
        norm_labels = torch.sum(labels * labels, dim=1)
        scalar_product = torch.sum(outputs * labels, dim=1)
        norm_product = torch.sqrt(norm_outputs * norm_labels)

        mask = norm_product == 0
        scalar_product = mask * self.eps + torch.logical_not(mask) * scalar_product
        norm_product = mask * self.eps + torch.logical_not(mask) * norm_product
        scalar_product = torch.flatten(scalar_product, 1, 2)
        norm_product = torch.flatten(norm_product, 1, 2)
        angle = torch.sum(scalar_product / norm_product, dim=1) / norm_product.shape[1]
        loss = 1 - angle
        return torch.mean(loss)
