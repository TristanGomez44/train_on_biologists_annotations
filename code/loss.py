
import sys 

import torch
from torch.nn import functional as F

from utils import remove_no_annot,make_class_nb_dic
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
        self.map_sim_term_weight= args.map_sim_term_weight
        print(args.map_sim_term_weight)

        self.distribution_learning = args.distribution_learning
        print(self.distribution_learning)

    def forward(self,target_dic,output_dict):
        return supervised_loss(target_dic, output_dict,self.regression,self.class_nb_targ_dic,self.plcc_weight,self.rank_weight,self.task_to_train,map_sim_term_weight=self.map_sim_term_weight,distr_learn=self.distribution_learning)

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
    target = target.float()
    p_loss, r_loss = plcc_loss(output, target), rank_loss(output, target)
    loss = plcc_weight*p_loss + rank_weight*r_loss
    return loss

def feat_norm(feat_maps):
    norm = torch.abs(feat_maps).sum(dim=1,keepdim=True).float()
    norm_min = norm.min(dim=2,keepdim=True)[0].min(dim=3,keepdim=True)[0]
    norm_max = norm.max(dim=2,keepdim=True)[0].max(dim=3,keepdim=True)[0]
    norm = (norm-norm_min)
    norm = norm/(norm_max-norm_min)
    return norm

def supervised_loss(target_dic, output_dict,regression,class_nb_targ_dic,plcc_weight,rank_weight,task_to_train,kl_temp=1,kl_interp=0.5,map_sim_term_weight=0,distr_learn=False):
    loss_dic = {}

    loss = 0

    for target_name,target in target_dic.items():

        if task_to_train=="all" or target_name==task_to_train:

            if "master_output_"+target_name in output_dict:
                output = output_dict["output_"+target_name]
                master_output = output_dict["master_output_"+target_name]
                kl = F.kl_div(F.log_softmax(output/kl_temp, dim=1),F.softmax(master_output/kl_temp, dim=1),reduction="sum")
                
                output,target_onlyannot = remove_no_annot(output_dict["output_"+target_name],target)
                ce = F.cross_entropy(output , target_onlyannot,reduction="sum")
                
                sub_loss = (kl*kl_interp*kl_temp*kl_temp+ce*(1-kl_interp))
                loss_dic[f"loss_{target_name}"] = sub_loss.data.unsqueeze(0)
                loss += sub_loss            
            else:
                    
                if regression:
                    output,target = remove_no_annot(output_dict["output_"+target_name],target) 
                    class_nb = class_nb_targ_dic[target_name]
                    sub_loss = regression_loss(output,target,class_nb,plcc_weight,rank_weight)
                else:
                    if distr_learn:
                        output = output_dict["output_"+target_name]
                        sub_loss =  F.kl_div(F.log_softmax(output,dim=-1),target,reduction="none")
                    else:
                        output,target = remove_no_annot(output_dict["output_"+target_name],target) 
                        sub_loss = F.cross_entropy(output, target,reduction="none")
            
                loss_dic[f"loss_{target_name}"] = sub_loss.data.sum().unsqueeze(0)
                loss += sub_loss.sum()

    if task_to_train == "all":
        loss /= 3

    if map_sim_term_weight>0:
        icm_map,te_map,exp_map = feat_norm(output_dict["feat_icm"]),feat_norm(output_dict["feat_te"]),feat_norm(output_dict["feat_exp"])
        sim_term = (icm_map*te_map).mean(dim=(1,2,3))+(icm_map*exp_map).mean(dim=(1,2,3))+(te_map*exp_map).mean(dim=(1,2,3))
        sim_term = (sim_term/3).sum()
        loss += map_sim_term_weight*sim_term

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
