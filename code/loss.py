
import sys 

import torch
from torch.nn import functional as F

from utils import remove_no_annot,_remove_no_annot
from grade_dataset import NO_ANNOT 

class Loss(torch.nn.Module):
    def __init__(self,args,reduction="sum"):
        super(Loss, self).__init__()
        self.reduction = reduction
        self.args= args
    def forward(self,target_dic,resDict):
        return computeLoss(self.args, target_dic, resDict,reduction=self.reduction)

def computeLoss(args,target_dic, resDict,reduction="sum"):
    loss_dic = {}

    loss = 0

    annot_nb_list = [target_dic[key] != NO_ANNOT for key in target_dic]
    annot_nb_list = torch.stack(annot_nb_list,axis=0).sum(axis=0)

    for target_name,target in target_dic.items():
        annot_nb_list_onlyannot = _remove_no_annot(annot_nb_list,target)
 
        output,target = remove_no_annot(resDict["output_"+target_name],target)
      
        loss_ce = F.cross_entropy(output, target,reduction="none")
        loss_dic[f"loss_{target_name}"] = loss_ce.data.sum().unsqueeze(0)

        loss += args.nll_weight*(loss_ce/annot_nb_list_onlyannot).sum()

    loss_dic["loss"] = loss.unsqueeze(0)

    return loss_dic

#From https://github.com/sthalles/SimCLR/
def info_nce_loss(features,n_views=2,temperature=0.07,reduction="sum",normalisation=True):
    batch_size = features.shape[0]//n_views

    labels = torch.cat([torch.arange(batch_size) for i in range(n_views)], dim=0)
    labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
    labels = labels.to(features.device)

    features = F.normalize(features, dim=1)

    similarity_matrix = torch.matmul(features, features.T)
    # assert similarity_matrix.shape == (
    #     n_views * batch_size, n_views * batch_size)
    # assert similarity_matrix.shape == labels.shape

    # discard the main diagonal from both: labels and similarities matrix
    mask = torch.eye(labels.shape[0], dtype=torch.bool).to(features.device)
    labels = labels[~mask].view(labels.shape[0], -1)
    similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
    # assert similarity_matrix.shape == labels.shape

    # select and combine multiple positives
    positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

    if normalisation:
        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(features.device)

        logits = logits / temperature
        return F.cross_entropy(logits, labels,reduction=reduction)
    else:

        if reduction == "sum":
            return -positives.sum()
        elif reduction == "mean":
            return -positives.mean()
        else:
            raise ValueError("Unkown reduction",reduction)


'''
Implementation of Focal Loss with adaptive gamma.
Reference:
[1]  T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollar, Focal loss for dense object detection.
     arXiv preprint arXiv:1708.02002, 2017.
'''
def get_gamma_dic():
    return {0.2:5.0,0.5:3.0,1:1}

def get_gamma_list(pt):
    gamma_dict = get_gamma_dic()
    gamma_list = []
    batch_size = pt.shape[0]
    for i in range(batch_size):
        pt_sample = pt[i].item()

        j = 0
        gamma_found =False
        thres_list = list(sorted(gamma_dict.keys()))
        while (not gamma_found) and (j < len(thres_list)):

            gamma = gamma_dict[thres_list[j]]
            gamma_found = pt_sample < thres_list[j]

            if gamma_found:
                gamma_list.append(gamma)
            else:
                j += 1
        
        if not gamma_found:
            gamma_list.append(gamma_dict[thres_list[-1]])

    return torch.tensor(gamma_list).to(pt.device)

def adaptive_focal_loss(logits, target,reduction):

    target = target.view(-1,1)
    logpt = F.log_softmax(logits, dim=1).gather(1,target).view(-1)
    pt = F.softmax(logits, dim=1).gather(1,target).view(-1)

    gamma = get_gamma_list(pt)
    loss = -1 * (1-pt)**gamma * logpt

    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError("Unkown reduction method",reduction)

def agregate_losses(loss_dic):
    for loss_name in loss_dic:
        loss_dic[loss_name] = loss_dic[loss_name].sum()
    return loss_dic