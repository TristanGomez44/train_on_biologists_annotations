
import sys 

import torch
from torch.nn import functional as F

from utils import remove_no_annot,_remove_no_annot,make_class_nb_dic
from grade_dataset import NO_ANNOT 

def agregate_losses(loss_dic):
    for loss_name in loss_dic:
        loss_dic[loss_name] = loss_dic[loss_name].sum()
    return loss_dic

class SupervisedLoss(torch.nn.Module):
    def __init__(self,regression=False,args=None):
        super().__init__()

        self.regression = regression
        self.class_nb_targ_dic= make_class_nb_dic(args)
        self.plcc_weight = args.plcc_weight
        self.rank_weight = args.rank_weight
        self.task_to_train = args.task_to_train

    def forward(self,target_dic,output_dict):
        return supervised_loss(target_dic, output_dict,self.regression,self.class_nb_targ_dic,self.plcc_weight,self.rank_weight,self.task_to_train)

def plcc_loss(y_pred, y):
    y_pred = y_pred.squeeze(1)
    sigma_hat, m_hat = torch.std_mean(y_pred, unbiased=False)
    y_pred = (y_pred - m_hat) / (sigma_hat + 1e-8)
    sigma, m = torch.std_mean(y, unbiased=False)
    y = (y - m) / (sigma + 1e-8)
    loss0 = torch.nn.functional.mse_loss(y_pred, y) / 4
    rho = torch.mean(y_pred * y)
    loss1 = torch.nn.functional.mse_loss(rho * y_pred, y) / 4
    return ((loss0 + loss1) / 2).float()

def rank_loss(y_pred, y):
    ranking_loss = torch.nn.functional.relu(
        (y_pred - y_pred.t()) * torch.sign((y.t() - y))
    )
    scale = 1 + torch.max(ranking_loss)
    return (
        torch.sum(ranking_loss) / y_pred.shape[0] / (y_pred.shape[0] - 1) / scale
    ).float()

def regression_loss(output,target,class_nb,plcc_weight,rank_weight):

    target = (target+1)/(class_nb+1)

    p_loss, r_loss = plcc_loss(output, target), rank_loss(output, target)
    loss = plcc_weight*p_loss + rank_weight*r_loss
    return loss

def supervised_loss(target_dic, output_dict,regression,class_nb_targ_dic,plcc_weight,rank_weight,task_to_train,kl_temp=1,kl_interp=0.5):
    loss_dic = {}

    loss = 0

    annot_nb_list = [target_dic[key] != NO_ANNOT for key in target_dic]
    annot_nb_list = torch.stack(annot_nb_list,axis=0).sum(axis=0)

    for target_name,target in target_dic.items():
        if task_to_train=="all" or target_name==task_to_train:
            annot_nb_list_onlyannot = _remove_no_annot(annot_nb_list,target)
    
            if "master_output_"+target_name:
                output,target_onlyannot = remove_no_annot(output_dict["output_"+target_name],target)
                master_output,target_onlyannot = remove_no_annot(output_dict["master_output_"+target_name],target)

                kl = F.kl_div(F.log_softmax(output/kl_temp, dim=1),F.softmax(master_output/kl_temp, dim=1),reduction="none")
                ce = F.cross_entropy(output, target_onlyannot,reduction="none")
                loss = (kl*kl_interp*kl_temp*kl_temp+ce*(1-kl_interp))

            else:
                output,target = remove_no_annot(output_dict["output_"+target_name],target)
                
                if regression:
                    class_nb = class_nb_targ_dic[target_name]
                    sub_loss = regression_loss(output,target,class_nb,plcc_weight,rank_weight)
                else:
                    sub_loss = F.cross_entropy(output, target,reduction="none")
                
                loss_dic[f"loss_{target_name}"] = sub_loss.data.sum().unsqueeze(0)

                loss += (sub_loss/annot_nb_list_onlyannot).sum()

    loss_dic["loss"] = loss.unsqueeze(0)

    return loss_dic

def cross_entropy(teacher_output,student_output,teacher_temp,student_temp,center):
    teacher_output = teacher_output.detach() # stop gradient
    teacher_output = F.softmax((teacher_output - center) / teacher_temp, dim=1) # center + sharpen
    student_log_output = F.log_softmax(student_output/student_temp,dim=1)
    return - (teacher_output * student_log_output).sum(dim=1).sum()

class SelfSuperVisedLoss(torch.nn.Module):
    def forward(self,student_dict,teacher_dict):
        return self_supervised_loss(student_dict,teacher_dict)


def self_supervised_loss(student_dict,teacher_dict):

    student_temp = student_dict["temp"]
    teacher_temp = teacher_dict["temp"]
    center = teacher_dict["center"]

    loss = cross_entropy(teacher_dict["output1"],student_dict["output2"],teacher_temp,student_temp,center)/2
    loss += cross_entropy(teacher_dict["output2"],student_dict["output1"],teacher_temp,student_temp,center)/2
    loss_dic = {"loss":loss.unsqueeze(0)}
    return loss_dic
