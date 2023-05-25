
import sys 

import torch
from torch.nn import functional as F

from utils import remove_no_annot,_remove_no_annot
from grade_dataset import NO_ANNOT 

class SupervisedLoss(torch.nn.Module):
    def forward(self,target_dic,output_dict):
        return supervised_loss(target_dic, output_dict)

class SelfSuperVisedLoss(torch.nn.Module):
    def forward(self,student_dict,teacher_dict):
        return self_supervised_loss(student_dict,teacher_dict)

def supervised_loss(target_dic, output_dict):
    loss_dic = {}

    loss = 0

    annot_nb_list = [target_dic[key] != NO_ANNOT for key in target_dic]
    annot_nb_list = torch.stack(annot_nb_list,axis=0).sum(axis=0)

    for target_name,target in target_dic.items():
        annot_nb_list_onlyannot = _remove_no_annot(annot_nb_list,target)
 
        output,target = remove_no_annot(output_dict["output_"+target_name],target)
      
        loss_ce = F.cross_entropy(output, target,reduction="none")
        loss_dic[f"loss_{target_name}"] = loss_ce.data.sum().unsqueeze(0)

        loss += (loss_ce/annot_nb_list_onlyannot).sum()

    loss_dic["loss"] = loss.unsqueeze(0)

    return loss_dic

def cross_entropy(teacher_output,student_output,teacher_temp,student_temp,center):
    teacher_output = teacher_output.detach() # stop gradient
    teacher_output = F.softmax((teacher_output - center) / teacher_temp, dim=1) # center + sharpen
    student_log_output = F.log_softmax(student_output/student_temp,dim=1)
    return - (teacher_output * student_log_output).sum(dim=1).sum()

def self_supervised_loss(student_dict,teacher_dict):

    student_temp = student_dict["temp"]
    teacher_temp = teacher_dict["temp"]
    center = teacher_dict["center"]

    loss = cross_entropy(teacher_dict["output1"],student_dict["output2"],teacher_temp,student_temp,center)/2
    loss += cross_entropy(teacher_dict["output2"],student_dict["output1"],teacher_temp,student_temp,center)/2
    loss_dic = {"loss":loss.unsqueeze(0)}
    return loss_dic

def agregate_losses(loss_dic):
    for loss_name in loss_dic:
        loss_dic[loss_name] = loss_dic[loss_name].sum()
    return loss_dic
