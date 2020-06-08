import utils
import numpy as np
import torch
import scipy as sp
import torch
import sys
import torch.nn.functional as F
import math

import load_data

def updateMetrDict(metrDict,metrDictSample):

    if metrDict is None:
        metrDict = metrDictSample
    else:
        for metric in metrDict.keys():
            metrDict[metric] += metrDictSample[metric]

    return metrDict

def binaryToMetrics(output,target,resDict):
    ''' Computes metrics over a batch of targets and predictions

    Args:
    - output (list): the batch of outputs
    - target (list): the batch of ground truth class
    - transition_matrix (torch.tensor) : this matrix contains at row i and column j the empirical probability to go from state i to j

    '''

    acc = compAccuracy(output,target)
    metDict = {"Accuracy":acc}

    cleanNames = ["Accuracy_aux","Accuracy_puretext","Accuracy_struct","Accuracy_zoom"]
    keys = ["auxPred","puretext_pred","struct_pred","pred_zoom"]
    for i in range(len(keys)):
        if keys[i] in resDict:
            metDict[cleanNames[i]] = compAccuracy(resDict[keys[i]],target)

    return metDict

def compAccuracy(output,target):
    pred = output.argmax(dim=-1)
    acc = (pred == target).float().sum()/(pred.numel())
    return acc
