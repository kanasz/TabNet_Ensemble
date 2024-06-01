import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

num_classes = 2


class LDAMLoss(nn.Module):

    def __init__(self, cls_num_list, max_m=0.5, weight=None, s=30, device='cuda'):
        super(LDAMLoss, self).__init__()
        m_list = 1.0 / np.sqrt(np.sqrt(cls_num_list))
        m_list = m_list * (max_m / np.max(m_list))
        m_list = torch.FloatTensor(m_list).to(torch.device('cuda')).float()
        self.m_list = m_list
        assert s > 0
        self.s = s
        self.weight = torch.cuda.FloatTensor(weight)

    def forward(self, x, target, features=None):
        index = torch.zeros_like(x, dtype=torch.uint8)
        index.scatter_(1, target.data.view(-1, 1), 1)

        index_float = index.type(torch.FloatTensor)
        batch_m = torch.matmul(self.m_list[None, :], index_float.transpose(0, 1).to(torch.device('cuda')))
        batch_m = batch_m.view((-1, 1))
        x_m = x - batch_m

        output = torch.where(index, x_m, x)
        loss = F.cross_entropy(self.s * output, target, weight=self.weight)
        return loss
