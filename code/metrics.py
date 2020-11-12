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

def binaryToMetrics(output,target,segmentation,resDict):
    ''' Computes metrics over a batch of targets and predictions

    Args:
    - output (list): the batch of outputs
    - target (list): the batch of ground truth class
    - transition_matrix (torch.tensor) : this matrix contains at row i and column j the empirical probability to go from state i to j

    '''

    acc = compAccuracy(output,target)
    metDict = {"Accuracy":acc}

    for key in resDict.keys():
        if key.find("pred_") != -1:
            suff = key.split("_")[-1]
            metDict["Accuracy_{}".format(suff)] = compAccuracy(resDict[key],target)

    if "attMaps" in resDict.keys():
        metDict["Sparsity"],metDict["Sparsity Normalised"] = compAttMapSparsity(resDict["attMaps"].clone(),segmentation.clone() if not segmentation is None else None)

        #if not segmentation is None:
        #    metDict["IoS"] = compIoS(resDict["attMaps"],segmentation,resDict,bilinear,clus)

    return metDict

def compAccuracy(output,target):
    pred = output.argmax(dim=-1)
    acc = (pred == target).float().sum()
    return acc.item()

def compAttMapSparsity(attMaps,segmentation):
    max = attMaps.max(dim=-1,keepdim=True)[0].max(dim=-2,keepdim=True)[0]
    attMaps = attMaps/(max+0.00001)

    if attMaps.size(1) > 1:
        attMaps = attMaps.sum(dim=1,keepdim=True)

    sparsity = attMaps.mean(dim=(2,3))

    if not segmentation is None:
        factor = segmentation.size(-1)/attMaps.size(-1)
        sparsity_norm = sparsity/((segmentation>0.5).sum(dim=(1,2,3))/factor).float()
    else:
        sparsity_norm = torch.zeros_like(sparsity)

    return sparsity.sum().item(),sparsity_norm.sum().item()

def compIoS(attentionMap,segmentation,resDict,bilinear,clus):

    segmentation = (segmentation>0.5)

    if (not bilinear) and (not clus):
        thresholds = [2500]
    elif bilinear and clus:
        thresholds = [0.5]
    else:
        thresholds = [0]

    if bilinear > 1:
        attentionMap = attentionMap.mean(dim=1,keepdim=True)
        norm = torch.sqrt(torch.pow(resDict["features"],2).sum(dim=1,keepdim=True))
        norm_max = norm.max(dim=-1,keepdim=True)[0].max(dim=-2,keepdim=True)[0]
        norm = norm/norm_max
        attentionMap = attentionMap*norm

    attentionMap = F.interpolate(attentionMap,size=(segmentation.size(-1)))

    allIos = []

    for thres in thresholds:
        num = ((attentionMap>thres)*segmentation[:,0:1]).sum(dim=(1,2,3)).float()
        denom = (attentionMap>thres).sum(dim=(1,2,3)).float()
        ios = num/denom
        ios[torch.isnan(ios)] = 0
        allIos.append(ios.unsqueeze(0))

    finalIos = torch.cat(allIos,dim=0).mean(dim=0).sum().item()
    return finalIos
