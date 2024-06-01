import torch
import torch.nn as nn
import torch.nn.functional as F

num_classes = 2


class CrossEntropyLoss(nn.Module):

    def __init__(self,  weight=None, device='cuda'):
        super(CrossEntropyLoss, self).__init__()
        self.device = device
        if self.device == 'cpu':
            self.weight = torch.FloatTensor(weight)
        else:
            self.weight = torch.cuda.FloatTensor(weight)

    def forward(self, x, target,features=None):
        softmax_pred = torch.nn.Softmax(dim=-1)(x.to(torch.float64))
        if self.device=='cpu':
            target = F.one_hot(target).to(torch.device('cpu')).float()  # Change to float here
        else:
            target = F.one_hot(target).to(torch.device('cuda')).float()  # Change to float here

        return F.cross_entropy(softmax_pred, target, weight=self.weight)
