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

def catIntermediateVariables(visualDict,intermVarDict,nbVideos):

    intermVarDict["fullAttMap"] = catMap(visualDict,intermVarDict["fullAttMap"],key="attMaps")
    intermVarDict["fullNormSeq"] = catMap(visualDict,intermVarDict["fullNormSeq"],key="norm")

    return intermVarDict

def saveIntermediateVariables(intermVarDict,exp_id,model_id,epoch,mode="val"):

    intermVarDict["fullAttMap"] = saveMap(intermVarDict["fullAttMap"],exp_id,model_id,epoch,mode,key="attMaps")
    intermVarDict["fullNormSeq"] = saveMap(intermVarDict["fullNormSeq"],exp_id,model_id,epoch,mode,key="norm")

    return intermVarDict

def catMap(visualDict,fullMap,key="attMaps"):
    if key in visualDict.keys():

        #In case attention weights are not comprised between 0 and 1
        tens_min = visualDict[key].min(dim=-1,keepdim=True)[0].min(dim=-2,keepdim=True)[0].min(dim=-3,keepdim=True)[0]
        tens_max = visualDict[key].max(dim=-1,keepdim=True)[0].max(dim=-2,keepdim=True)[0].max(dim=-3,keepdim=True)[0]
        map = (visualDict[key]-tens_min)/(tens_max-tens_min)

        if fullMap is None:
            fullMap = (map.cpu()*255).byte()
        else:
            fullMap = torch.cat((fullMap,(map.cpu()*255).byte()),dim=0)

    return fullMap

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
