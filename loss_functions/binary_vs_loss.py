import torch
import torch.nn as nn
import torch.nn.functional as F

num_classes = 2


class BinaryVSLoss(nn.Module):

    def __init__(self, iota_pos=0.0, iota_neg=0.0, Delta_pos=1.0, Delta_neg=1.0, weight=None, device='cuda'):
        super(BinaryVSLoss, self).__init__()
        self.device = device
        if self.device == 'cpu':
            iota_list = torch.tensor([iota_neg, iota_pos]).to(torch.device('cpu')).float()
            Delta_list = torch.tensor([Delta_neg, Delta_pos]).to(torch.device('cpu')).float()
            self.iota_list = torch.FloatTensor(iota_list).to(torch.device('cpu'))
            self.Delta_list = torch.FloatTensor(Delta_list).to(torch.device('cpu'))
            self.weight = torch.FloatTensor(weight)
        else:
            iota_list = torch.tensor([iota_neg, iota_pos]).to(torch.device('cuda')).float()
            Delta_list = torch.tensor([Delta_neg, Delta_pos]).to(torch.device('cuda')).float()
            self.iota_list = torch.cuda.FloatTensor(iota_list)
            self.Delta_list = torch.cuda.FloatTensor(Delta_list)
            self.weight = torch.cuda.FloatTensor(weight)

    def forward(self, x, target,features):
        index = torch.zeros((x.shape[0], 2), dtype=torch.uint8)
        softmax_pred = torch.nn.Softmax(dim=-1)(x.to(torch.float64))
        if self.device=='cpu':
            target = F.one_hot(target).to(torch.device('cpu')).float()  # Change to float here
            index_float = index.type(torch.FloatTensor)
        else:
            target = F.one_hot(target).to(torch.device('cuda')).float()  # Change to float here
            index_float = index.type(torch.cuda.FloatTensor)
        index_float.scatter_(1, target.long(), 1)
        batch_iota = torch.matmul(self.iota_list, index_float.t())
        batch_Delta = torch.matmul(self.Delta_list, index_float.t())

        batch_iota = batch_iota.view((-1, 1))
        batch_Delta = batch_Delta.view((-1, 1))

        output = softmax_pred * batch_Delta - batch_iota
        #output = x * batch_Delta - batch_iota
        return F.binary_cross_entropy_with_logits(30 * output, target, weight=self.weight)
