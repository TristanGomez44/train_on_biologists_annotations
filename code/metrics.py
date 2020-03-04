import utils
import numpy as np
import torch
import scipy as sp
import torch
import sys
import torch.nn.functional as F
import math

import load_data

# Code taken from https://gist.github.com/PetrochukM/afaa3613a99a8e7213d2efdd02ae4762#file-top_k_viterbi-py-L5
# Credits to AllenNLP for the base implementation and base tests:
# https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py#L174

# Modified AllenNLP `viterbi_decode` to support `top_k` sequences efficiently.
def emptyMetrDict():
    return {"Loss":0,"Accuracy":0}

def updateMetrDict(metrDict,metrDictSample):

    for metric in metrDict.keys():

        if metric in list(metrDictSample.keys()):
            if metric.find("Entropy") == -1:
                metrDict[metric] += metrDictSample[metric]
            else:
                if metrDict[metric] is None:
                    metrDict[metric] = metrDictSample[metric]
                else:
                    metrDict[metric] = torch.cat((metrDict[metric],metrDictSample[metric]),dim=0)

    return metrDict

def binaryToMetrics(output,target):
    ''' Computes metrics over a batch of targets and predictions

    Args:
    - output (list): the batch of outputs
    - target (list): the batch of ground truth class
    - transition_matrix (torch.tensor) : this matrix contains at row i and column j the empirical probability to go from state i to j

    '''

    #Simple Accuracy
    pred = output.argmax(dim=-1)

    acc = (pred == target).float().sum()/(pred.numel())

    metDict = {"Accuracy":acc}

    return metDict
