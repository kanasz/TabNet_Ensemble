import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from loss_functions.ib_loss import ib_loss

num_classes = 2


def ib_focal_loss(input_values, ib, gamma):
    """Computes the ib focal loss"""
    p = torch.exp(-input_values)
    loss = (1 - p) ** gamma * input_values * ib
    return loss.mean()


class IBFocalLoss(nn.Module):
    def __init__(self, weight=None, alpha=10000.,epsilon=0.001, gamma=0., device='cuda'):
        super(IBFocalLoss, self).__init__()
        assert alpha > 0

        #self.alpha = alpha
        #self.epsilon = epsilon
        #self.weight = weight
        #self.gamma = gamma
        self.device = device
        if self.device == 'cpu':
            self.weight = torch.FloatTensor(weight)
            self.epsilon = torch.FloatTensor([epsilon]).to(torch.device('cpu')).float()
            self.alpha = torch.FloatTensor([alpha]).to(torch.device('cpu')).float()
            self.gamma = torch.FloatTensor([gamma]).to(torch.device('cpu')).float()
        else:
            self.weight = torch.cuda.FloatTensor(weight)
            self.epsilon = torch.cuda.FloatTensor([epsilon])
            self.alpha = torch.cuda.FloatTensor([alpha])
            self.gamma = torch.cuda.FloatTensor([gamma])

    def forward(self, input, target, features):
        #grads = torch.sum(torch.abs(F.softmax(input, dim=1) - F.one_hot(target, num_classes)), 1)  # N * 1
        #features = torch.sum(features, dim=1) / features.shape[1]
        #ib = grads * (features.reshape(-1))
        #ib = self.alpha / (ib + self.epsilon)
        #return ib_focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), ib, self.gamma)
        grads = torch.sum(torch.abs(F.softmax(input, dim=1) - F.one_hot(target, num_classes)), 1)  # N * 1
        features = torch.sum(features, dim=1) / features.shape[1]
        ib = grads * features.reshape(-1)
        ib = self.alpha / (ib + self.epsilon)
        result = ib_focal_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), ib, self.gamma)

        return result
