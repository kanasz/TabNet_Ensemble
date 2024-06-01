import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

num_classes = 2


def ib_loss(input_values, ib):
    """Computes the focal loss"""
    loss = input_values * ib
    return loss.mean()


class IBLossMDR(nn.Module):
    def __init__(self, weight=None, alpha=10000., epsilon=0.001, l =0, device='cuda'):
        super(IBLossMDR, self).__init__()
        self.device = device
        assert alpha > 0
        self.l = l
        if self.device == 'cpu':
            self.weight = torch.FloatTensor(weight).to(torch.device('cpu')).float()
            self.epsilon = torch.FloatTensor([epsilon]).to(torch.device('cpu')).float()
            self.alpha = torch.FloatTensor([alpha]).to(torch.device('cpu')).float()
        else:
            self.weight = torch.cuda.FloatTensor(weight)
            self.epsilon = torch.cuda.FloatTensor([epsilon])
            self.alpha = torch.cuda.FloatTensor([alpha])

    def forward(self, input, target, features):
        grads = torch.sum(torch.abs(F.softmax(input, dim=1) - F.one_hot(target, num_classes)), 1)  # N * 1
        features = torch.sum(features,dim=1)/features.shape[1]
        ib = grads * features.reshape(-1)
        ib = self.alpha / (ib + self.epsilon)
        loss = ib_loss(F.cross_entropy(input, target, reduction='none', weight=self.weight), ib)
        softmax_pred = torch.nn.Softmax(dim=-1)(input.to(torch.float64))
        pred_class = torch.argmax(softmax_pred, dim=1)
        loss = loss + (self.l / 2) * (((1 / 2) - torch.mean(pred_class.to(torch.float64))) ** 2)
        return loss