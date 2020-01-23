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
def viterbi_decode(tag_sequence,transition_matrix,top_k=1):
    """
    Perform Viterbi decoding in log space over a sequence given a transition matrix
    specifying pairwise (transition) potentials between tags and a matrix of shape
    (sequence_length, num_tags) specifying unary potentials for possible tags per
    timestep.
    Parameters
    ----------
    tag_sequence : torch.Tensor, required.
        A tensor of shape (sequence_length, num_tags) representing scores for
        a set of tags over a given sequence.
    transition_matrix : torch.Tensor, required.
        A tensor of shape (num_tags, num_tags) representing the binary potentials
        for transitioning between a given pair of tags.
    top_k : int, required.
        Integer defining the top number of paths to decode.
    Returns
    -------
    viterbi_path : List[int]
        The tag indices of the maximum likelihood tag sequence.
    viterbi_score : float
        The score of the viterbi path.
    """

    transition_matrix = transition_matrix.to(tag_sequence.device)

    sequence_length, num_tags = list(tag_sequence.size())

    path_scores = []
    path_indices = []
    # At the beginning, the maximum number of permutations is 1; therefore, we unsqueeze(0)
    # to allow for 1 permutation.
    path_scores.append(tag_sequence[0, :].unsqueeze(0))
    # assert path_scores[0].size() == (n_permutations, num_tags)

    # Evaluate the scores for all possible paths.
    for timestep in range(1, sequence_length):
        # Add pairwise potentials to current scores.
        # assert path_scores[timestep - 1].size() == (n_permutations, num_tags)
        summed_potentials = path_scores[timestep - 1].unsqueeze(2) + transition_matrix
        summed_potentials = summed_potentials.view(-1, num_tags)

        # Best pairwise potential path score from the previous timestep.
        max_k = min(summed_potentials.size()[0], top_k)
        scores, paths = torch.topk(summed_potentials, k=max_k, dim=0)
        # assert scores.size() == (n_permutations, num_tags)
        # assert paths.size() == (n_permutations, num_tags)

        scores = tag_sequence[timestep, :] + scores
        # assert scores.size() == (n_permutations, num_tags)
        path_scores.append(scores)
        path_indices.append(paths.squeeze())

    # Construct the most likely sequence backwards.
    path_scores = path_scores[-1].view(-1)
    max_k = min(path_scores.size()[0], top_k)
    viterbi_scores, best_paths = torch.topk(path_scores, k=max_k, dim=0)
    viterbi_paths = []
    for i in range(max_k):
        viterbi_path = [best_paths[i]]
        for backward_timestep in reversed(path_indices):
            viterbi_path.append(int(backward_timestep.view(-1)[viterbi_path[-1]]))
        # Reverse the backward path.
        viterbi_path.reverse()
        # Viterbi paths uses (num_tags * n_permutations) nodes; therefore, we need to modulo.
        viterbi_path = [j % num_tags for j in viterbi_path]
        viterbi_paths.append(viterbi_path)
    return viterbi_paths, viterbi_scores

def emptyMetrDict():
    return {"Loss":0,"Accuracy":0,"Accuracy (Viterbi)":0,"Correlation":0,"Temp Accuracy":0}

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

def binaryToMetrics(output,target,transition_matrix,videoNames=None,onlyPairsCorrelation=True):
    ''' Computes metrics over a batch of targets and predictions

    Args:
    - output (list): the batch of outputs
    - target (list): the batch of ground truth class
    - transition_matrix (torch.tensor) : this matrix contains at row i and column j the empirical probability to go from state i to j

    '''

    #Simple Accuracy
    pred = output.argmax(dim=-1)

    acc = (pred == target).float().sum()/(pred.numel())

    if torch.isnan(transition_matrix).sum() == 0:

        #Accuracy with viterbi
        pred = []
        for outSeq in output:

            outSeq = torch.nn.functional.softmax(outSeq, dim=-1)

            predSeqs,_ = viterbi_decode(torch.log(outSeq),torch.log(transition_matrix),top_k=1)

            pred.append(torch.tensor(predSeqs[0]).unsqueeze(0))

        pred = torch.cat(pred,dim=0).to(target.device)

        accViterb = (pred == target).float().sum()/(pred.numel())

    else:
        accViterb = 0

    metDict = {"Accuracy":acc,'Accuracy (Viterbi)':accViterb}

    if not videoNames is None:
        metDict["Correlation"],metDict["Temp Accuracy"] = correlation(pred,target,videoNames,onlyPairs=onlyPairsCorrelation)

    return metDict

def correlation(predBatch,target,videoNames,onlyPairs=True):
    ''' Computes the times at which the model predicts the developpement phase is changing and
    compare it to the real times where the phase is changing. Computes a correlation between those
    two list of numbers.

    '''

    for i,pred in enumerate(predBatch):

        dataset = load_data.getDataset(videoNames[i])
        timeElapsedTensor = np.genfromtxt("../data/{}/annotations/{}_timeElapsed.csv".format(dataset,videoNames[i]),delimiter=",")[1:,1]

        phasesPredDict = phaseToTime(pred,timeElapsedTensor)
        phasesTargDict = phaseToTime(target[0],timeElapsedTensor)

        commonPhases = list(set(list(phasesPredDict.keys())).intersection(set(list(phasesTargDict.keys()))))
        timePairs = []
        accuracy = 0
        for phase in commonPhases:
            timePairs.append((phasesPredDict[phase],phasesTargDict[phase]))
            if np.abs(phasesPredDict[phase]-phasesTargDict[phase]) <= 1:
                accuracy +=1
        accuracy /= len(phasesTargDict.keys())

        if onlyPairs:
            return timePairs,accuracy
        else:
            timePairs = np.array(timePairs)
            return np.corrcoef(timePairs[:,0],timePairs[:,1])[0,1],accuracy

def phaseToTime(phaseList,timeElapsedTensor):
    changingPhaseFrame = np.concatenate(([1],(phaseList[1:]-phaseList[:-1]) > 0),axis=0)
    phases = phaseList[np.argwhere(changingPhaseFrame)[:,0]]

    changingPhaseFrame = np.argwhere(changingPhaseFrame)[:,0]
    changingPhaseTime = timeElapsedTensor[changingPhaseFrame]

    phaseToFrameDict = {phases[i].item():changingPhaseTime[i] for i in range(len(phases))}

    return phaseToFrameDict
