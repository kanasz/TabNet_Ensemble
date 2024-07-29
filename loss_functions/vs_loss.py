import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

num_classes = 2

class VSLoss(nn.Module):

    def __init__(self, cls_num_list, gamma=0.3, tau=1.0, weight=None, device='cuda'):
        super(VSLoss, self).__init__()

        cls_probs = [cls_num / sum(cls_num_list) for cls_num in cls_num_list]
        temp = (1.0 / np.array(cls_num_list)) ** gamma
        temp = temp / np.min(temp)

        iota_list = tau * np.log(cls_probs)
        Delta_list = temp

        self.iota_list = torch.cuda.FloatTensor(iota_list)
        self.Delta_list = torch.cuda.FloatTensor(Delta_list)
        self.weight = torch.cuda.FloatTensor(weight)

    def forward(self, x, target, features=None):
        softmax_pred = torch.nn.Softmax(dim=-1)(x.to(torch.float64))

        target = F.one_hot(target).float()  # Change to float here
        #output = x / self.Delta_list + self.iota_list
        output = softmax_pred / self.Delta_list + self.iota_list
        loss = F.cross_entropy(output, target)
        #print(loss)
        return loss