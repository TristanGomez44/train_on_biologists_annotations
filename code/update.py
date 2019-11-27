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

def computeScore(model,allFeats,timeElapsed,allTarget,valLTemp,vidName):

    allOutput = {"allPred":None}
    splitSizes = [valLTemp for _ in range(allFeats.size(1)//valLTemp)]

    if allFeats.size(1)%valLTemp > 0:
        splitSizes.append(allFeats.size(1)%valLTemp)

    chunkList = torch.split(allFeats,split_size_or_sections=splitSizes,dim=1)

    timeElapsedChunkList = torch.split(timeElapsed,split_size_or_sections=splitSizes,dim=1)

    sumSize = 0

    for i in range(len(chunkList)):

        output = model.tempModel(chunkList[i].squeeze(0),batchSize=1,timeTensor=timeElapsedChunkList[i])

        for tensorName in output.keys():
            if not tensorName in allOutput.keys():
                allOutput[tensorName] = output[tensorName]
            else:
                allOutput[tensorName] = torch.cat((allOutput[tensorName],output[tensorName]),dim=1)

        sumSize += len(chunkList[i])

    return allOutput

def updateMetrics(args,model,allFeat,timeElapsed,allTarget,precVidName,nbVideos,metrDict,outDict,targDict,attMapDict=None):

    allOutputDict = computeScore(model,allFeat,timeElapsed,allTarget,args.val_l_temp,precVidName)

    allOutput = allOutputDict["pred"]

    if args.compute_metrics_during_eval:
        if args.regression:
            #Converting the output of the sigmoid between 0 and 1 to a scale between -0.5 and class_nb+0.5
            allOutput = (torch.sigmoid(allOutput)*(args.class_nb+1)-0.5)
            loss = F.mse_loss(allOutput,allTarget.float())
        elif args.uncertainty:
            loss = 0
        else:
            loss = F.cross_entropy(allOutput.squeeze(0),allTarget.squeeze(0)).data.item()

        metDictSample = metrics.binaryToMetrics(allOutput,allTarget,model.transMat,args.regression,args.uncertainty)
        metDictSample["Loss"] = loss
        metrDict = metrics.updateMetrDict(metrDict,metDictSample)
        if args.regression:
            allOutput = metrics.regressionPred2Confidence(allOutput,args.class_nb)

    outDict[precVidName] = allOutput
    targDict[precVidName] = allTarget

    if not attMapDict is None:
        attMapDict[precVidName] = allOutputDict["attention"]

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
def updateLR(epoch,maxEpoch,lr,startEpoch,kwargsOpti,kwargsTr,lrCounter,net,optimConst):
    #This condition determines when the learning rate should be updated (to follow the learning rate schedule)
    #The optimiser have to be rebuilt every time the learning rate is updated
    if (epoch-1) % ((maxEpoch + 1)//len(lr)) == 0 or epoch==startEpoch:

        kwargsOpti['lr'] = lr[lrCounter]
        optim = optimConst(net.parameters(), **kwargsOpti)

        kwargsTr["optim"] = optim

        if lrCounter<len(lr)-1:
            lrCounter += 1

    return kwargsOpti,kwargsTr,lrCounter
