from torch.nn import functional as F
import metrics
import trainVal
import numpy as np
import load_data
import torch
import matplotlib.cm as cm
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from sklearn.metrics import roc_auc_score
import utils
import sys
import subprocess
import psutil
import os

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

def computeScore(model,allFeats,timeElapsed,allTarget,valLTemp,vidName):

    allOutput = {"allPred":None}
    splitSizes = [valLTemp for _ in range(allFeats.size(1)//valLTemp)]

    if allFeats.size(1)%valLTemp > 0:
        splitSizes.append(allFeats.size(1)%valLTemp)

    chunkList = torch.split(allFeats,split_size_or_sections=splitSizes,dim=1)

    timeElapsedChunkList = torch.split(timeElapsed,split_size_or_sections=splitSizes,dim=1)

    sumSize = 0

    for i in range(len(chunkList)):

        output = model.secondModel(chunkList[i].squeeze(0),batchSize=1,timeTensor=timeElapsedChunkList[i])

        for tensorName in output.keys():
            if not tensorName in allOutput.keys():
                allOutput[tensorName] = output[tensorName]
            else:
                allOutput[tensorName] = torch.cat((allOutput[tensorName],output[tensorName]),dim=1)

        sumSize += len(chunkList[i])

    return allOutput

def updateMetrics(args,model,allFeat,timeElapsed,allTarget,precVidName,nbVideos,metrDict,outDict,targDict):

    allOutputDict = computeScore(model,allFeat,timeElapsed,allTarget,args.val_l_temp,precVidName)

    allOutput = allOutputDict["pred"]

    if args.compute_metrics_during_eval:
        loss = F.cross_entropy(allOutput.squeeze(0),allTarget.squeeze(0)).data.item()

        metDictSample = metrics.binaryToMetrics(allOutput,allTarget,model.transMat,videoMode=True)
        metDictSample["Loss"] = loss
        metrDict = metrics.updateMetrDict(metrDict,metDictSample)

    outDict[precVidName] = allOutput
    targDict[precVidName] = allTarget

    nbVideos += 1

    return allOutput,nbVideos

def updateFrameDict(frameIndDict,frameInds,vidName):
    ''' Store the prediction of a model in a dictionnary with one entry per movie

    Args:
     - outDict (dict): the dictionnary where the scores will be stored
     - output (torch.tensor): the output batch of the model
     - frameIndDict (dict): a dictionnary collecting the index of each frame used
     - vidName (str): the name of the video from which the score are produced

    '''

    if vidName in frameIndDict.keys():
        reshFrInds = frameInds.view(len(frameInds),-1).clone()
        frameIndDict[vidName] = torch.cat((frameIndDict[vidName],reshFrInds),dim=0)

    else:
        frameIndDict[vidName] = frameInds.view(len(frameInds),-1).clone()
def updateLR_and_Optim(epoch,maxEpoch,lr,startEpoch,kwargsOpti,kwargsTr,lrCounter,net,optimConst):
    #This condition determines when the learning rate should be updated (to follow the learning rate schedule)
    #The optimiser have to be rebuilt every time the learning rate is updated
    if (epoch-1) % ((maxEpoch + 1)//len(lr)) == 0 or epoch==startEpoch:

        kwargsOpti['lr'] = lr[lrCounter]
        optim = optimConst(net.parameters(), **kwargsOpti)

        kwargsTr["optim"] = optim

        if lrCounter<len(lr)-1:
            lrCounter += 1

    return kwargsOpti,kwargsTr,lrCounter
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
            updateOccupiedGPURamCSV(epoch,"train",exp_id,model_id,batch_idx)
        updateOccupiedRamCSV(epoch,"train",exp_id,model_id,batch_idx)
        updateOccupiedCPUCSV(epoch,"train",exp_id,model_id,batch_idx)
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
def updateOccupiedCPUCSV(epoch,mode,exp_id,model_id,batch_idx):

    cpuOccList = psutil.cpu_percent(percpu=True)

    csvPath = "../results/{}/{}_cpuLoad_{}.csv".format(exp_id,model_id,mode)

    if epoch==1 and batch_idx==0:
        with open(csvPath,"w") as text_file:
            print("epoch,"+",".join([str(i) for i in range(len(cpuOccList))]),file=text_file)
            print(str(epoch)+","+",".join([str(cpuOcc) for cpuOcc in cpuOccList]),file=text_file)
    else:
        with open(csvPath,"a") as text_file:
            print(str(epoch)+","+",".join([str(cpuOcc) for cpuOcc in cpuOccList]),file=text_file)
def updateOccupiedRamCSV(epoch,mode,exp_id,model_id,batch_idx):

    ramOcc = psutil.virtual_memory()._asdict()["percent"]

    csvPath = "../results/{}/{}_occCPURam_{}.csv".format(exp_id,model_id,mode)

    if epoch==1 and batch_idx==0:
        with open(csvPath,"w") as text_file:
            print("epoch,"+","+"percent",file=text_file)
            print(str(epoch)+","+str(ramOcc),file=text_file)
    else:
        with open(csvPath,"a") as text_file:
            print(str(epoch)+","+str(ramOcc),file=text_file)
def updateTimeCSV(epoch,mode,exp_id,model_id,totalTime,batch_idx):

    csvPath = "../results/{}/{}_time_{}.csv".format(exp_id,model_id,mode)

    if epoch==1 and batch_idx==0:
        with open(csvPath,"w") as text_file:
            print("epoch,"+","+"time",file=text_file)
            print(str(epoch)+","+str(totalTime),file=text_file)
    else:
        with open(csvPath,"a") as text_file:
            print(str(epoch)+","+str(totalTime),file=text_file)

def catIntermediateVariables(visualDict,intermVarDict,nbVideos):

    intermVarDict["fullAttMap"] = catMap(visualDict,intermVarDict["fullAttMap"],key="attMaps")
    intermVarDict["fullFeatMapSeq"] = catMap(visualDict,intermVarDict["fullFeatMapSeq"],key="features")
    intermVarDict["fullAffTransSeq"] = catAffineTransf(visualDict,intermVarDict["fullAffTransSeq"])
    intermVarDict["fullPointsSeq"] = catPointsSeq(visualDict,intermVarDict["fullPointsSeq"])
    if nbVideos < 6:
        intermVarDict["fullReconstSeq"] = catMap(visualDict,intermVarDict["fullReconstSeq"],key="reconst")

    return intermVarDict
def saveIntermediateVariables(intermVarDict,exp_id,model_id,epoch,precVidName=""):

    intermVarDict["fullAttMap"] = saveMap(intermVarDict["fullAttMap"],exp_id,model_id,epoch,precVidName,key="attMaps")
    intermVarDict["fullFeatMapSeq"] = saveMap(intermVarDict["fullFeatMapSeq"],exp_id,model_id,epoch,precVidName,key="featMaps")
    intermVarDict["fullAffTransSeq"] = saveAffineTransf(intermVarDict["fullAffTransSeq"],exp_id,model_id,epoch,precVidName)
    intermVarDict["fullPointsSeq"] =  savePointsSeq(intermVarDict["fullPointsSeq"],exp_id,model_id,epoch,precVidName)
    intermVarDict["fullReconstSeq"] = saveMap(intermVarDict["fullReconstSeq"],exp_id,model_id,epoch,precVidName,key="reconst")

    return intermVarDict
def catAffineTransf(visualDict,fullAffTransSeq):
    if "theta" in visualDict.keys():
        if fullAffTransSeq is None:
            fullAffTransSeq = visualDict["theta"].cpu()
        else:
            fullAffTransSeq = torch.cat((fullAffTransSeq,visualDict["theta"].cpu()),dim=0)

    return fullAffTransSeq
def saveAffineTransf(fullAffTransSeq,exp_id,model_id,epoch,precVidName):
    if not fullAffTransSeq is None:
        np.save("../results/{}/affTransf_{}_epoch{}_{}.npy".format(exp_id,model_id,epoch,precVidName),fullAffTransSeq.numpy())
        fullAffTransSeq = None
    return fullAffTransSeq
def catPointsSeq(visualDict,fullPointsSeq):
    if "points" in visualDict.keys():
        if fullPointsSeq is None:
            fullPointsSeq = visualDict["points"].cpu()
        else:
            fullPointsSeq = torch.cat((fullPointsSeq,visualDict["points"].cpu()),dim=0)

    return fullPointsSeq
def savePointsSeq(fullPointsSeq,exp_id,model_id,epoch,precVidName):
    if not fullPointsSeq is None:
        np.save("../results/{}/points_{}_epoch{}_{}.npy".format(exp_id,model_id,epoch,precVidName),fullPointsSeq.numpy())
        fullPointsSeq = None
    return fullPointsSeq
def catMap(visualDict,fullMap,key="attMaps"):
    if key in visualDict.keys():

        if not type(visualDict[key]) is dict:
            if key == "features" or key == "reconst":
                visualDict[key] = (visualDict[key]-visualDict[key].min())/(visualDict[key].max()-visualDict[key].min())

            if fullMap is None:
                fullMap = (visualDict[key].cpu()*255).byte()
            else:
                fullMap = torch.cat((fullMap,(visualDict[key].cpu()*255).byte()),dim=0)

        else:
            visualDict[key] = {layer:(visualDict[key][layer].cpu()*255).byte() for layer in visualDict[key].keys()}

            if fullMap is None:
                fullMap = visualDict[key]
            else:
                for layer in fullMap.keys():
                    fullMap[layer] = torch.cat((fullMap[layer],visualDict[key][layer]),dim=0)

    return fullMap
def saveMap(fullMap,exp_id,model_id,epoch,precVidName,key="attMaps"):
    if not fullMap is None:

        if not type(fullMap) is dict:
            np.save("../results/{}/{}_{}_epoch{}_{}.npy".format(exp_id,key,model_id,epoch,precVidName),fullMap.numpy())
        else:
            for layer in fullMap.keys():
                np.save("../results/{}/{}_{}_epoch{}_{}_lay{}.npy".format(exp_id,key,model_id,epoch,precVidName,layer),fullMap[layer].numpy())
        fullMap = None
    return fullMap
