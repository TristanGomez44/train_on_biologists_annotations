import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import math

import separability_study 

#Keys for ECE metric 
COUNT = 'count'
CONF = 'conf'
ACC = 'acc'
BIN_ACC = 'bin_acc'
BIN_CONF = 'bin_conf'

def add_losses_to_dic(metDictSample,lossDic):
    for loss_name in lossDic:
        metDictSample[loss_name] = lossDic[loss_name].item()
    return metDictSample

def updateMetrDict(metrDict,metrDictSample):

    if metrDict is None:
        metrDict = metrDictSample
    else:
        for metric in metrDict.keys():
            metrDict[metric] += metrDictSample[metric]

    return metrDict

def binaryToMetrics(output,target,resDict,comp_spars=False):
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

    if "attMaps" in resDict.keys() and comp_spars:
        spar = compAttMapSparsity(resDict["attMaps"].clone(),resDict["feat"].clone())
        metDict["Sparsity"] = spar
    else:
        norm = torch.sqrt(torch.pow(resDict["feat"],2).sum(dim=1,keepdim=True))
        spar = compSparsity(norm)
        metDict["Sparsity"] = spar 

    return metDict

def separability_metric(feat_pooled,feat_pooled_masked,target_list,metDict,seed,nb_per_class):

    label_to_ind = {}
    for i in range(len(feat_pooled)):
        lab = target_list[i].item()
        if lab not in label_to_ind:
            label_to_ind[lab] = []  
        label_to_ind[lab].append(i)

    torch.manual_seed(seed)
    kept_inds = []
    for label in label_to_ind:
        all_inds = torch.tensor(label_to_ind[label])
        all_inds_perm = all_inds[torch.randperm(len(all_inds))]
        kept_inds.extend(all_inds_perm[:nb_per_class])
    kept_inds = torch.tensor(kept_inds)
    
    feat_pooled,feat_pooled_masked = feat_pooled[kept_inds],feat_pooled_masked[kept_inds]

    sep_dict = separability_study.run_separability_analysis(feat_pooled,feat_pooled_masked,False,seed)
    separability_auc,separability_acc = sep_dict["val_auc"].mean(),sep_dict["val_acc"].mean()
    metDict["Sep_AuC"] = separability_auc
    metDict["Sep_Acc"] = separability_acc
    return metDict

def compAccuracy(output,target):
    pred = output.argmax(dim=-1)
    acc = (pred == target).float().sum()
    return acc.item()

def compSparsity(norm):
    norm_max = norm.max(dim=-1,keepdim=True)[0].max(dim=-2,keepdim=True)[0]
    norm = norm/(norm_max+0.00001)
    sparsity = norm.mean(dim=(2,3))
    return sparsity.sum().item()

def compAttMapSparsity(attMaps,features=None):
    if not features is None:
        norm = torch.sqrt(torch.pow(features,2).sum(dim=1,keepdim=True))
        norm_max = norm.max(dim=-1,keepdim=True)[0].max(dim=-2,keepdim=True)[0]
        norm = norm/norm_max

        attMaps = attMaps*norm

    max_val = attMaps.max(dim=-1,keepdim=True)[0].max(dim=-2,keepdim=True)[0]
    attMaps = attMaps/(max_val+0.00001)

    if attMaps.size(1) > 1:
        attMaps = attMaps.mean(dim=1,keepdim=True)

    sparsity = attMaps.mean(dim=(2,3))

    return sparsity.sum().item()

def comptAttMapSparN(sparsity,segmentation,attMaps):

    factor = segmentation.size(-1)/attMaps.size(-1)
    sparsity_norm = sparsity/((segmentation>0.5).sum(dim=(2,3)).sum(dim=1,keepdim=True)/factor).float()
    return sparsity_norm.sum().item()

def compIoS(attMapNorm,segmentation):

    segmentation = (segmentation>0.5)

    thresholds = torch.arange(10)*1.0/10

    attMapNorm = F.interpolate(attMapNorm,size=(segmentation.size(-1)),mode="bilinear",align_corners=False)

    allIos = []

    for thres in thresholds:
        num = ((attMapNorm>thres)*segmentation[:,0:1]).sum(dim=(1,2,3)).float()
        denom = (attMapNorm>thres).sum(dim=(1,2,3)).float()
        ios = num/denom
        ios[torch.isnan(ios)] = 0
        allIos.append(ios.unsqueeze(0))

    finalIos = torch.cat(allIos,dim=0).mean(dim=0)
    return finalIos.sum().item()


'''
Metrics to measure calibration of a trained deep neural network.
References:
[1] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger. On calibration of modern neural networks.
    arXiv preprint arXiv:1706.04599, 2017.
'''
def _bin_initializer(bin_dict, num_bins=10):
    for i in range(num_bins):
        bin_dict[i][COUNT] = 0
        bin_dict[i][CONF] = 0
        bin_dict[i][ACC] = 0
        bin_dict[i][BIN_ACC] = 0
        bin_dict[i][BIN_CONF] = 0

def _populate_bins(confs, preds, labels, num_bins=10):
    bin_dict = {}
    for i in range(num_bins):
        bin_dict[i] = {}
    _bin_initializer(bin_dict, num_bins)
    num_test_samples = len(confs)

    for i in range(0, num_test_samples):
        confidence = confs[i]
        prediction = preds[i]
        label = labels[i]
        binn = int(math.ceil(((num_bins * confidence) - 1)))
        bin_dict[binn][COUNT] = bin_dict[binn][COUNT] + 1
        bin_dict[binn][CONF] = bin_dict[binn][CONF] + confidence
        bin_dict[binn][ACC] = bin_dict[binn][ACC] + \
            (1 if (label == prediction) else 0)

    for binn in range(0, num_bins):
        if (bin_dict[binn][COUNT] == 0):
            bin_dict[binn][BIN_ACC] = 0
            bin_dict[binn][BIN_CONF] = 0
        else:
            bin_dict[binn][BIN_ACC] = float(
                bin_dict[binn][ACC]) / bin_dict[binn][COUNT]
            bin_dict[binn][BIN_CONF] = bin_dict[binn][CONF] / \
                float(bin_dict[binn][COUNT])
    return bin_dict

def expected_calibration_error(all_outputs, all_target, metrDict,num_bins=10):

    all_outputs = F.softmax(all_outputs,dim=-1)
    preds = all_outputs.argmax(dim=-1)
    preds = preds.view(-1,1)
    confs = all_outputs.gather(1,preds)
 
    bin_dict = _populate_bins(confs, preds, all_target, num_bins)
    num_samples = len(all_target)
    ece = 0
    for i in range(num_bins):
        bin_accuracy = bin_dict[i][BIN_ACC]
        bin_confidence = bin_dict[i][BIN_CONF]
        bin_count = bin_dict[i][COUNT]
        ece += (float(bin_count) / num_samples) * \
            abs(bin_accuracy - bin_confidence)

    metrDict["ECE"] = ece.item()
    
    return metrDict

def main():

    example_nb = 2000

    feat_pooled = torch.rand(size=(example_nb,2048))
    feat_pooled_masked = torch.rand(size=(example_nb,2048))

    class_nb = 8
    nb_per_class = 15

    target_list = torch.arange(class_nb).unsqueeze(-1)
    target_list = target_list.expand(-1,example_nb//class_nb).reshape(-1)

    metDict = {}
    seed = 0

    retDict = separability_metric(feat_pooled,feat_pooled_masked,target_list,metDict,seed,nb_per_class)

    print(retDict)

if __name__ == "__main__":

    main()
