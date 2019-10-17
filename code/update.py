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

def computeScore(model,allFeats,allTarget,valLTemp,vidName):

    allOutput = None
    splitSizes = [valLTemp for _ in range(allFeats.size(1)//valLTemp)]

    if allFeats.size(1)%valLTemp > 0:
        splitSizes.append(allFeats.size(1)%valLTemp)

    chunkList = torch.split(allFeats,split_size_or_sections=splitSizes,dim=1)

    sumSize = 0

    for i in range(len(chunkList)):

        output = model.computeScore(chunkList[i])

        if allOutput is None:
            allOutput = output
        else:
            allOutput = torch.cat((allOutput,output),dim=1)

        sumSize += len(chunkList[i])

    return allOutput

def updateMetrics(args,model,allFeat,allTarget,precVidName,nbVideos,metrDict,outDict,targDict):
    ''' Update the current estimation

    Also compute the scene change scores if the temporal model is a CNN

    '''

    allOutput = computeScore(model,allFeat,allTarget,args.val_l_temp,precVidName)

    outDict[precVidName] = allOutput
    targDict[precVidName] = allTarget

    if args.compute_val_metrics:
        loss = F.cross_entropy(allOutput.squeeze(0),allTarget.squeeze(0)).data.item()
        metrDict["Loss"] += loss
        allPred = allOutput.view(allOutput.size(0)*allOutput.size(1),allOutput.size(2)).argmax(dim=-1)
        metrDict["Accuracy"] += metrics.binaryToMetrics(allPred,allTarget)["Accuracy"]

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
