from torch.nn import functional as F
import numpy as np
import torch
import matplotlib.cm as cm
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.metrics import roc_auc_score
import subprocess
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

def updateHardWareOccupation(debug,benchmark,cuda,epoch,mode,exp_id,model_id,batch_idx):
    if debug or benchmark:
        if cuda:
            updateOccupiedGPURamCSV(epoch,mode,exp_id,model_id,batch_idx)
        updateOccupiedRamCSV(epoch,mode,exp_id,model_id,batch_idx)
        updateOccupiedCPUCSV(epoch,mode,exp_id,model_id,batch_idx)

def updateOccupiedGPURamCSV(epoch,mode,exp_id,model_id,batch_idx):

    occRamDict = get_gpu_memory_map()

    csvPath = "../results/{}/{}_occRam_{}.csv".format(exp_id,model_id,mode)

    if epoch==1 and batch_idx==0:
        with open(csvPath,"w") as text_file:
            print("epoch,"+",".join([str(device) for device in occRamDict.keys()]),file=text_file)
            print(str(epoch)+","+",".join([occRamDict[device] for device in occRamDict.keys()]),file=text_file)
    else:
        with open(csvPath,"a") as text_file:
            print(str(epoch)+","+",".join([occRamDict[device] for device in occRamDict.keys()]),file=text_file)

def updateTimeCSV(epoch,mode,exp_id,model_id,totalTime,batch_idx):

    csvPath = "../results/{}/{}_time_{}.csv".format(exp_id,model_id,mode)

    if epoch==1 and batch_idx==0:
        with open(csvPath,"w") as text_file:
            print("epoch,"+","+"time",file=text_file)
            print(str(epoch)+","+str(totalTime),file=text_file)
    else:
        with open(csvPath,"a") as text_file:
            print(str(epoch)+","+str(totalTime),file=text_file)

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

def catIntermediateVariables(resDict,intermVarDict):

    intermVarDict = catMap(resDict,intermVarDict,"attMaps")
    intermVarDict = catMap(resDict,intermVarDict,"norm")

    if "feat_pooled_masked" in resDict:
        for key in ["feat_pooled","feat_pooled_masked"]:
            intermVarDict = cat(resDict[key],key,intermVarDict)

    return intermVarDict

def catFeat(resDict,featDic):
    for key in ["feat_pooled","feat_pooled_masked"]:
        featDic = cat(resDict[key],key,featDic)
    return featDic

def cat(tensor,key,intermVarDict):
    if not key in intermVarDict:
        intermVarDict[key] = tensor
    else:
        intermVarDict[key] = torch.cat((intermVarDict[key],tensor),dim=0)
    return intermVarDict

def catMap(resDict,intermVarDict,key="attMaps"):
    if key in resDict.keys():

        #In case attention weights are not comprised between 0 and 1
        tens_min = resDict[key].min(dim=-1,keepdim=True)[0].min(dim=-2,keepdim=True)[0].min(dim=-3,keepdim=True)[0]
        tens_max = resDict[key].max(dim=-1,keepdim=True)[0].max(dim=-2,keepdim=True)[0].max(dim=-3,keepdim=True)[0]
        maps = (resDict[key]-tens_min)/(tens_max-tens_min)
        maps = (maps.cpu()*255).byte()

        intermVarDict = cat(maps,key,intermVarDict)

    return intermVarDict

def saveIntermediateVariables(intermVarDict,exp_id,model_id,epoch,mode="val"):
    if "attMaps" in intermVarDict:
        intermVarDict["attMaps"] = saveMap(intermVarDict["attMaps"],exp_id,model_id,epoch,mode,key="attMaps")
    intermVarDict["norm"] = saveMap(intermVarDict["norm"],exp_id,model_id,epoch,mode,key="norm")

    return intermVarDict

def saveMap(fullMap,exp_id,model_id,epoch,mode,key="attMaps"):
    if not fullMap is None:
        np.save("../results/{}/{}_{}_epoch{}_{}.npy".format(exp_id,key,model_id,epoch,mode),fullMap.numpy())
        fullMap = None
    return fullMap

def get_gpu_memory_map():
    """Get the current gpu usage.

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used','--format=csv,nounits,noheader'], encoding='utf-8')
    # Convert lines into a dictionary
    gpu_memory = [x for x in result.strip().split('\n')]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map

class NCEWeightUpdater():

    def __init__(self,args,threshold_epoch_nb=5,variation_threshold=1e-2,increase_factor=0.5):
        self.args = args
        self.threshold_epoch_nb = threshold_epoch_nb
        self.variation_threshold = variation_threshold
        self.increase_factor = increase_factor
        self.loss_history_dic = {}

    def init_nce_weight(self):
        self.args.nce_weight = self.args.nce_sched_start
        return self.args.nce_weight

    def update_nce_weight(self,metrDict):

        loss_names = list(filter(lambda x:x.find("loss") != -1,list(metrDict.keys())))

        for loss_name in loss_names:
            if not loss_name in self.loss_history_dic:
                self.loss_history_dic[loss_name] = []
            
            self.loss_history_dic[loss_name].append(metrDict[loss_name])

        converged_list = []

        for loss_name in self.loss_history_dic.keys():

            values = self.loss_history_dic[loss_name]

            if len(values) > self.threshold_epoch_nb:

                last_values = np.array(values[-self.threshold_epoch_nb-1:])
                
                #variations = np.abs(last_values[:-1] - last_values[1:])/last_values[:-1]
                #criterion = (variations<self.variation_threshold).all()

                criterion = (last_values[-1] - last_values[:-1] < 0).all()

                converged_list.append(criterion)

            else:
                converged_list.append(False)

        converged_list = np.array(converged_list)

        if converged_list.all():
            self.args.nce_weight *= 1+self.increase_factor  
        
        return self.args.nce_weight