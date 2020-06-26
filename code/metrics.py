import utils
import numpy as np
import torch
import scipy as sp
import torch
import sys
import torch.nn.functional as F
import math

import load_data
import torchvision

import matplotlib.pyplot as plt
plt.switch_backend('agg')

def updateMetrDict(metrDict,metrDictSample):

    if metrDict is None:
        metrDict = metrDictSample
    else:
        for metric in metrDict.keys():
            metrDict[metric] += metrDictSample[metric]

    return metrDict

def binaryToMetrics(output,target,segmentation,resDict,normAtt=False,attentionAct="sigmoid"):
    ''' Computes metrics over a batch of targets and predictions

    Args:
    - output (list): the batch of outputs
    - target (list): the batch of ground truth class
    - transition_matrix (torch.tensor) : this matrix contains at row i and column j the empirical probability to go from state i to j

    '''

    acc = compAccuracy(output,target)
    metDict = {"Accuracy":acc}

    cleanNames = ["Accuracy_aux","Accuracy_puretext","Accuracy_struct","Accuracy_zoom","Accuracy_crop","Accuracy_drop","Accuracy_rawcrop"]
    keys = ["auxPred","puretext_pred","struct_pred","pred_zoom","pred_crop","pred_drop","pred_rawcrop"]
    for i in range(len(keys)):
        if keys[i] in resDict:
            metDict[cleanNames[i]] = compAccuracy(resDict[keys[i]],target)

    if "predBilClusEns0" in resDict:
        i = 0
        key = "predBilClusEns{}".format(i)
        while key in resDict:
            metDict["Accuracy_BilClustEns{}".format(i)] = compAccuracy(resDict[key],target)
            i += 1
            key = "predBilClusEns{}".format(i)

    if "attMaps" in resDict.keys():
        metDict["Sparsity"] = compAttMapSparsity(resDict["attMaps"].clone())
        if not segmentation is None:
            metDict["IoU"] = compIoU(resDict["attMaps"],segmentation,normAtt,attentionAct)

    return metDict

def compAccuracy(output,target):
    pred = output.argmax(dim=-1)
    acc = (pred == target).float().sum()
    return acc.item()

def compAttMapSparsity(attMaps):
    max = attMaps.max(dim=-1,keepdim=True)[0].max(dim=-2,keepdim=True)[0]
    attMaps = attMaps/(max+0.00001)
    return attMaps.mean(dim=(2,3)).sum().item()

def compIoU(attentionMap,segmentation,normAtt,attentionAct):

    segmentation = (segmentation>0.5)

    if normAtt:
        thresholds = [2500]
    else:
        if attentionAct == "softmax":
            thresholds = [0.5]
        elif attentionAct == "relu":
            thresholds = [0]
        elif attentionAct == "sigmoid":
            thresholds = [0.5]
        else:
            raise ValueError("Unkown activation function :",attentionAct)

    attentionMap = F.interpolate(attentionMap,size=(segmentation.size(-1)))

    allIou = []

    for thres in thresholds:
        iou = ((attentionMap>thres)*segmentation).sum(dim=(1,2,3)).float()/((attentionMap>thres) | segmentation).sum(dim=(1,2,3)).float()
        allIou.append(iou.unsqueeze(0))

    finalIou = torch.cat(allIou,dim=0).mean(dim=0).sum().item()
    return finalIou
