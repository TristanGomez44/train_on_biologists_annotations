import numpy as np
import torch
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os,glob 
import utils

def ssl_updates(args,optim,epoch):
    args.teach_momentum = schedule(epoch,args.start_teach_momentum,args.end_teach_momentum,args.epochs,mode="cosine")
    args.teach_temp = schedule(epoch,args.start_teach_temp,args.end_teach_temp,args.teach_temp_sched_epochs,mode="linear")
    args.weight_decay = schedule(epoch,args.start_weight_decay,args.end_weight_decay,args.epochs,mode="cosine")
    for i, param_group in enumerate(optim.param_groups):
        if i == 0:  # only the first group is regularized
            param_group["weight_decay"] = args.weight_decay
    return args,optim 

def update_center(center,teacher_dict,momentum):
    out1,out2 = teacher_dict["output1"],teacher_dict["output2"]
    out = torch.cat([out1,out2],dim=0)
    center = momentum*center+(1-momentum)*out.mean(dim=0,keepdim=True)
    return center 

def update_teacher(teacher_net,student_net,momentum):
    for teach_params, student_params in zip(teacher_net.parameters(), student_net.parameters()):       
        teach_params.data = momentum*teach_params.data+(1-momentum)*student_params.data
    return teacher_net

def schedule(epoch,start_value,end_value_temp,epoch_nb,mode="linear"):

    progress = (epoch-1)/(epoch_nb-1)

    if epoch <= epoch_nb:
        if mode == "linear":
            return start_value+(end_value_temp-start_value)*progress
        elif mode == "cosine":
            return end_value_temp + 0.5 * (start_value - end_value_temp) * (1 + np.cos(np.pi * progress))
        else:
            raise ValueError("Unkown schedule mode:",mode)
    else:
        return end_value_temp
    
def updateBestModel(metricVal,bestMetricVal,exp_id,model_id,bestEpoch,epoch,net,isBetter,worseEpochNb,args):

    if isBetter(metricVal,bestMetricVal):
        if os.path.exists("../models/{}/model{}_best_epoch{}".format(exp_id,model_id,bestEpoch)):
            os.remove("../models/{}/model{}_best_epoch{}".format(exp_id,model_id,bestEpoch))

        state_dic = net.module.state_dict() if args.swa else net.state_dict()

        torch.save(state_dic, "../models/{}/model{}_best_epoch{}".format(exp_id,model_id, epoch))
        bestEpoch = epoch
        bestMetricVal = metricVal
        worseEpochNb = 0
    else:
        worseEpochNb += 1

    return bestEpoch,bestMetricVal,worseEpochNb

def updateSeedAndNote(args):
    if args.start_mode == "auto" and (not args.optuna) and len(
            glob.glob("../models/{}/model{}_epoch*".format(args.exp_id, args.model_id))) > 0:
        args.seed += 1
        init_path = args.init_path
        if init_path == "None" and args.strict_init:
            init_path = sorted(glob.glob("../models/{}/model{}_epoch*".format(args.exp_id, args.model_id)),
                               key=utils.findLastNumbers)[-1]
        startEpoch = utils.findLastNumbers(init_path)
        args.note += ";s{} at {}".format(args.seed, startEpoch)
    return args

def all_cat_var_dic(var_dic,resDict,mode,save_output_during_validation=False):
    # Other variables produced by the net
    if mode == "test":
        if "feat" in resDict:
            norm = torch.sqrt(torch.pow(resDict["feat"],2).sum(dim=1,keepdim=True))
            var_dic = cat_var_dic(var_dic,"norm",norm)
    
        if "feat_pooled_per_head" in resDict:
            norm = torch.sqrt(torch.pow(resDict["feat_pooled_per_head"],2).sum(dim=2,keepdim=True))
            var_dic = cat_var_dic(var_dic,"norm_pooled_per_head",norm)

        for key in ["attMaps","feat_pooled"]:
            if key in resDict:
                var_dic = cat_var_dic(var_dic,key,resDict[key])
        
        for output_name in resDict:
            if "output" in output_name:
                var_dic = cat_var_dic(var_dic,output_name,resDict[output_name])

    return var_dic

def cat_var_dic(var_dic,tensor_name,tensor):
    
    assert tensor.ndim in [2,3,4,5]

    if tensor.ndim >= 4:
        preproc_func = preproc_maps 
    else:
        preproc_func = preproc_vect

    tensor = preproc_func(tensor)

    if not tensor_name in var_dic:
        var_dic[tensor_name] = tensor
    else:
        var_dic[tensor_name] = torch.cat((var_dic[tensor_name],tensor),dim=0)

    return var_dic

def preproc_maps(maps):
    if len(maps.shape) > 4:
        dim_tuple = (-1,-2,-3,-4)
    else:
        dim_tuple = (-1,-2,-3)
    maps_min = maps.amin(dim=dim_tuple,keepdim=True)[0]
    maps_max = maps.amax(dim=dim_tuple,keepdim=True)[0]
    maps = (maps-maps_min)/(maps_max-maps_min)
    maps = (maps.cpu()*255).byte()
    return maps

def preproc_vect(vect):
    return vect.detach().cpu()

def save_variables(intermVarDict,exp_id,model_id,epoch,mode="val"):

    key_list = ["attMaps","norm","norm_pooled_per_head"]
    for key in intermVarDict:
        if "output" in key:
            key_list.append(key)
    
    for key in key_list:
        if key in intermVarDict:
            print("savevar",key)
            save_variable(intermVarDict[key],exp_id,model_id,epoch,mode,key=key)

def save_variable(fullMap,exp_id,model_id,epoch,mode,key="attMaps"):
    np.save(f"../results/{exp_id}/{key}_{model_id}_epoch{epoch}_{mode}.npy",fullMap.numpy())