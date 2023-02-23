import numpy as np
import torch
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import os,glob 
import utils

def updateBestModel(metricVal,bestMetricVal,exp_id,model_id,bestEpoch,epoch,net,isBetter,worseEpochNb):

    if isBetter(metricVal,bestMetricVal):
        if os.path.exists("../models/{}/model{}_best_epoch{}".format(exp_id,model_id,bestEpoch)):
            os.remove("../models/{}/model{}_best_epoch{}".format(exp_id,model_id,bestEpoch))

        torch.save(net.state_dict(), "../models/{}/model{}_best_epoch{}".format(exp_id,model_id, epoch))
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

def all_cat_var_dic(var_dic,resDict,target,args,mode):
    # Other variables produced by the net
    if mode == "test":
        norm = torch.sqrt(torch.pow(resDict["feat"],2).sum(dim=1,keepdim=True))
        if "attMaps" in resDict:
            var_dic = cat_var_dic(var_dic,"attMaps",resDict["attMaps"])
        var_dic = cat_var_dic(var_dic,"norm",norm)

    if args.nce_weight > 0 or args.adv_weight > 0: 
        var_dic = cat_var_dic(var_dic,"feat_pooled_masked",resDict["feat_pooled_masked"])
        var_dic = cat_var_dic(var_dic,"feat_pooled",resDict["feat_pooled"])

    if args.focal_weight > 0:
        var_dic = cat_var_dic(var_dic,"output",resDict["output"])

    if args.focal_weight > 0 or args.nce_weight > 0 or args.adv_weight > 0:
        var_dic = cat_var_dic(var_dic,"target",target)

    return var_dic

def cat_var_dic(var_dic,tensor_name,tensor):
    
    assert tensor.ndim in [1,2,4]

    if tensor.ndim == 4:
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
    maps = maps.detach()
    maps_min = maps.min(dim=-1,keepdim=True)[0].min(dim=-2,keepdim=True)[0].min(dim=-3,keepdim=True)[0]
    maps_max = maps.max(dim=-1,keepdim=True)[0].max(dim=-2,keepdim=True)[0].max(dim=-3,keepdim=True)[0]
    maps = (maps-maps_min)/(maps_max-maps_max)
    maps = (maps.cpu()*255).byte()
    return maps

def preproc_vect(vect):
    return vect.detach().cpu()

def save_maps(intermVarDict,exp_id,model_id,epoch,mode="val"):
    if "attMaps" in intermVarDict:
        saveMap(intermVarDict["attMaps"],exp_id,model_id,epoch,mode,key="attMaps")
    saveMap(intermVarDict["norm"],exp_id,model_id,epoch,mode,key="norm")

def saveMap(fullMap,exp_id,model_id,epoch,mode,key="attMaps"):
    np.save("../results/{}/{}_{}_epoch{}_{}.npy".format(exp_id,key,model_id,epoch,mode),fullMap.numpy())

class NCEWeightUpdater():

    def __init__(self,args,epoch_nb=150):
        self.args = args
        self.epoch_nb = epoch_nb

    def compute_nce_weight(self,epoch):
        if epoch < self.epoch_nb:
            self.nce_weight = (epoch*1.0/self.epoch_nb)
        else:
            self.nce_weight = 1
        return self.nce_weight