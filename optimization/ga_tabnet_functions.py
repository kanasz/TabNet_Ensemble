import random

import numpy as np
import torch
from imblearn.metrics import geometric_mean_score

from constants import LossFunction, genes_ib_loss, genes_vs_loss, genes_ldam_loss, genes_focal_loss, \
    genes_ib_focal_loss, genes_binary_vs_loss, genes_vs_loss_mdr, genes_ldam_loss_mdr, genes_ib_loss_mdr, \
    genes_binary_vs_loss_mdr, genes_crossentropy_loss, genes_boosting_ib_loss, genes_boosting_vs_loss, \
    genes_boosting_ldam_loss, genes_boosting_focal_loss, genes_boosting_ib_focal_loss, genes_boosting_binary_vs_loss, \
    genes_boosting_vs_loss_mdr, genes_boosting_ldam_loss_mdr, genes_boosting_ib_loss_mdr, \
    genes_boosting_binary_vs_loss_mdr, genes_boosting_crossentropy_loss
# from loss_functions.ib_loss import LDAMLoss, BinaryVSLoss, VSLoss, IBLoss
from loss_functions.binary_vs_loss import BinaryVSLoss
from loss_functions.binary_vs_loss_mdr import BinaryVSLossMDR
from loss_functions.cross_entropy_loss import CrossEntropyLoss
from loss_functions.ib_focal_loss import IBFocalLoss
from loss_functions.ib_loss import IBLoss
from loss_functions.ib_loss_mdr import IBLossMDR
from loss_functions.ldam_loss import LDAMLoss
from loss_functions.ldam_loss_mdr import LDAMLossMDR
from loss_functions.vs_loss import VSLoss
from loss_functions.vs_loss_mdr import VSLossMDR
from pytorch_tabnet.metrics import Metric

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.rand(seed)
random.SystemRandom(seed)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # for multiGPUs.
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
#warnings.filterwarnings("ignore")




def get_gene_type_and_space(loss_function):
    if loss_function == LossFunction.IBLOSS:
        return genes_ib_loss
    if loss_function == LossFunction.VSLOSS:
        return genes_vs_loss
    if loss_function == LossFunction.LDAMLOSS:
        return genes_ldam_loss
    if loss_function == LossFunction.FOCALLOSS:
        return genes_focal_loss
    if loss_function == LossFunction.IBFOCALLOSS:
        return genes_ib_focal_loss
    if loss_function == LossFunction.BINARYVSLOSS:
        return genes_binary_vs_loss
    if loss_function == LossFunction.VSLOSSMDR:
        return genes_vs_loss_mdr
    if loss_function == LossFunction.LDAMLOSSMDR:
        return genes_ldam_loss_mdr
    if loss_function == LossFunction.IBLOSSMDR:
        return genes_ib_loss_mdr
    if loss_function == LossFunction.BINARYVSLOSSMDR:
        return genes_binary_vs_loss_mdr
    if loss_function ==LossFunction.CROSSENTROPYLOSS:
        return genes_crossentropy_loss
    return

def get_boosting_gene_type_and_space(loss_function):
    if loss_function == LossFunction.IBLOSS:
        return genes_boosting_ib_loss
    if loss_function == LossFunction.VSLOSS:
        return genes_boosting_vs_loss
    if loss_function == LossFunction.LDAMLOSS:
        return genes_boosting_ldam_loss
    if loss_function == LossFunction.FOCALLOSS:
        return genes_boosting_focal_loss
    if loss_function == LossFunction.IBFOCALLOSS:
        return genes_boosting_ib_focal_loss
    if loss_function == LossFunction.BINARYVSLOSS:
        return genes_boosting_binary_vs_loss
    if loss_function == LossFunction.VSLOSSMDR:
        return genes_boosting_vs_loss_mdr
    if loss_function == LossFunction.LDAMLOSSMDR:
        return genes_boosting_ldam_loss_mdr
    if loss_function == LossFunction.IBLOSSMDR:
        return genes_boosting_ib_loss_mdr
    if loss_function == LossFunction.BINARYVSLOSSMDR:
        return genes_boosting_binary_vs_loss_mdr
    if loss_function ==LossFunction.CROSSENTROPYLOSS:
        return genes_boosting_crossentropy_loss
    return


def get_loss(loss_function, params, cls_num_list, device):
    if loss_function == LossFunction.BINARYVSLOSS:
        return BinaryVSLoss(iota_pos=params[0], iota_neg=params[1], Delta_pos=params[2], Delta_neg=params[3],
                            weight=[params[4], params[5]],device=device)
    if loss_function == LossFunction.VSLOSS:
        return VSLoss(cls_num_list, gamma=params[0],tau=params[1], weight=[params[2], params[3]],device=device)
    if loss_function == LossFunction.IBLOSS:
        return IBLoss(weight=[params[0], params[1]], alpha=params[2], epsilon=params[3],device=device)
    if loss_function == LossFunction.IBFOCALLOSS:
        return IBFocalLoss(weight=[params[0], params[1]], alpha=params[2], epsilon=params[3], gamma=params[4], device=device)
    if loss_function == LossFunction.LDAMLOSS:
        return LDAMLoss(cls_num_list=cls_num_list,  weight=[params[1], params[2]], max_m=params[0], s=params[3], device=device)
    if loss_function == LossFunction.LDAMLOSS:
        return LDAMLoss(cls_num_list=cls_num_list,  weight=[params[1], params[2]], max_m=params[0], s=params[3], device=device)
    if loss_function == LossFunction.VSLOSSMDR:
        return VSLossMDR(cls_num_list, gamma=params[0],tau=params[1], weight=[params[2], params[3]], l=params[4], device=device)
    if loss_function == LossFunction.LDAMLOSSMDR:
        return LDAMLossMDR(cls_num_list=cls_num_list, weight=[params[1], params[2]], max_m=params[0], s=params[3], l=params[4],
                        device=device)
    if loss_function == LossFunction.IBLOSSMDR:
        return IBLossMDR(weight=[params[0], params[1]], alpha=params[2], epsilon=params[3], l=params[4], device=device)
    if loss_function == LossFunction.BINARYVSLOSSMDR:
        return BinaryVSLossMDR(iota_pos=params[0], iota_neg=params[1], Delta_pos=params[2], Delta_neg=params[3],
                            weight=[params[4], params[5]], l=params[6], device=device)
    if loss_function == LossFunction.CROSSENTROPYLOSS:
        return CrossEntropyLoss(weight=[params[0], params[1]])
    return


class GMean(Metric):
    def __init__(self):
        self._name = "GMean"
        self._maximize = True

    def __call__(self, y_true, y_score):
        y_pred = np.where(y_score[:, 1] > 0.5, 1, 0)
        gmean = geometric_mean_score(y_true, y_pred)
        return gmean


def lmse_loss(c_FP, c_TP, c_TN, c_FN):
    def my_loss_fn(y_pred, y_true):
        # logarithmic means squared error
        softmax_pred = torch.nn.Softmax(dim=-1)(y_pred.to(torch.float64))

        # logloss = (1-y_true)*torch.log(softmax_pred[:,0]) * c_FP
        # logloss = logloss + y_true*torch.log(softmax_pred[:,1]) * c_FN
        # logloss = logloss + (1-y_true) * torch.log(softmax_pred[:, 1]) * c_TN
        # logloss = logloss + y_true * torch.log(softmax_pred[:, 0]) * c_TP
        # res = -torch.mean(logloss)

        # max = torch.max(y_true - softmax_pred[:,0])
        # logloss = torch.log(1 - ((y_true - softmax_pred[:,0]) / max))
        # res2 = -torch.mean(logloss)
        # torch.save(y_pred, 'predicted.pt')
        # torch.save(y_true, 'true.pt')
        logloss = torch.pow(torch.log(1 + y_true) - torch.log(1 + (softmax_pred[:, 1])), 2)
        res2 = torch.mean(logloss)
        return res2

    return my_loss_fn


