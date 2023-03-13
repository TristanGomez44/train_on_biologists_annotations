import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import math
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn import svm
 
#Keys for ECE metric 
COUNT = 'count'
CONF = 'conf'
ACC = 'acc'
BIN_ACC = 'bin_acc'
BIN_CONF = 'bin_conf'

from saliency_maps_metrics.multi_step_metrics import Deletion, Insertion
from saliency_maps_metrics.single_step_metrics import IIC_AD, ADD

is_multi_step_dic = {"Deletion":True,"Insertion":True,"IIC_AD":False,"ADD":False}
const_dic = {"Deletion":Deletion,"Insertion":Insertion,"IIC_AD":IIC_AD,"ADD":ADD}

def get_sal_metric_dics():
    return is_multi_step_dic,const_dic

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
        if key.find("output_") != -1:
            suff = key.split("_")[-1]

            if not "adv" in key:
                metDict["Accuracy_{}".format(suff)] = compAccuracy(resDict[key],target)

    if "output_adv" in resDict:
        sigm_output_adv = torch.softmax(resDict["output_adv"].detach(),dim=1).cpu().numpy()[:,1]
        target_adv = resDict["target_adv"].detach().cpu().numpy()
        metDict["AuC_adv"] = roc_auc_score(target_adv,sigm_output_adv)*len(sigm_output_adv)*0.5
        metDict["Accuracy_adv"] = compAccuracy(resDict["output_adv"],resDict["target_adv"])*0.5

    if "attMaps" in resDict.keys() and comp_spars:
        spar = compAttMapSparsity(resDict["attMaps"].clone(),resDict["feat"].clone())
        metDict["Sparsity"] = spar

    else:
        norm = torch.sqrt(torch.pow(resDict["feat"],2).sum(dim=1,keepdim=True))
        spar = compSparsity(norm)
        metDict["Sparsity"] = spar 

    return metDict

def separability_metric(feat_pooled,feat_pooled_masked,label_list,metDict,seed,nb_per_class):

    kept_inds = sample_img_inds(nb_per_class,label_list=label_list) 

    feat_pooled,feat_pooled_masked = feat_pooled[kept_inds],feat_pooled_masked[kept_inds]

    sep_dict = run_separability_analysis(feat_pooled,feat_pooled_masked,False,seed)
    separability_auc,separability_acc = sep_dict["val_auc"].mean(),sep_dict["val_acc"].mean()
    metDict["Sep_AuC"] = separability_auc
    metDict["Sep_Acc"] = separability_acc

    return metDict

def saliency_metric_validity(testDataset,model,args,metDict,img_nb=20):

    nb_per_class = int(round(float(img_nb)/testDataset.num_classes))
    if nb_per_class == 0:
        nb_per_class = 1

    kept_inds = sample_img_inds(nb_per_class,testDataset=testDataset) 
    data,_ = getBatch(testDataset,kept_inds,args)
    net_lambda = lambda x:torch.softmax(model(x)["output"],dim=-1)
    
    resDict = model(data)
    predClassInds = resDict["output"].argmax(dim=-1)
    explanations = torch.sqrt(torch.pow(resDict["feat"],2).sum(dim=1,keepdim=True))
    
    is_multi_step_dic,const_dic = get_sal_metric_dics()
    
    for metric_name in const_dic:
 
        if not is_multi_step_dic[metric_name]:
            metric = const_dic[metric_name]()
            scores,scores_masked = metric.compute_scores(net_lambda,data,explanations,predClassInds)
            val_rate = add_validity_rate_single_step(metric_name,scores,scores_masked) 
            metDict[metric_name+"_val_rate"] = val_rate

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

def expected_calibration_error(var_dic,metrDict):

    metrDict["ECE"] = _expected_calibration_error(var_dic["output"], var_dic["target"])

    if "output_masked" in var_dic:
        metrDict["ECE_masked"] = _expected_calibration_error(var_dic["output_masked"], var_dic["target"])
   
    return metrDict

def _expected_calibration_error(all_outputs, all_target,num_bins=10):

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

    return ece.item()

def make_label_list(dataset):
    return [dataset.image_label[img_ind] for img_ind in sorted(dataset.image_label.keys())]

def sample_img_inds(nb_per_class,label_list=None,testDataset=None):

    assert (label_list is not None) or (testDataset is not None)
    
    if label_list is None:
        label_list = make_label_list(testDataset)

    label_to_ind = {}
    for i in range(len(label_list)):
        lab = label_list[i]

        if type(lab) is torch.Tensor:
            lab = lab.item()

        if lab not in label_to_ind:
            label_to_ind[lab] = []  
        label_to_ind[lab].append(i)

    torch.manual_seed(0)
    chosen_inds = []
    for label in label_to_ind:
        all_inds = torch.tensor(label_to_ind[label])
        all_inds_perm = all_inds[torch.randperm(len(all_inds))]
        chosen_class_inds = all_inds_perm[:nb_per_class]
        if len(chosen_class_inds) < nb_per_class:
            raise ValueError(f"Number of image to be sampled per class is too high for class {label} which has only {len(chosen_class_inds)} images.")
        chosen_inds.extend(chosen_class_inds)
    
    chosen_inds = torch.tensor(chosen_inds)
    
    return chosen_inds 

def applyPostHoc(attrFunc,data,targ,kwargs,args):

    if args.att_metrics_post_hoc.find("var") == -1 and args.att_metrics_post_hoc.find("smooth") == -1:
        argList = [data,targ]
    else:
        argList = [data]
        kwargs["target"] = targ

    attMap = attrFunc(*argList,**kwargs).clone().detach().to(data.device)

    if len(attMap.size()) == 2:
        attMap = attMap.unsqueeze(0).unsqueeze(0)
    elif len(attMap.size()) == 3:
        attMap = attMap.unsqueeze(0)
        
    return attMap

def getBatch(testDataset,inds,args):
    data_list = []
    targ_list = []
    for i in inds:
        batch = testDataset.__getitem__(i)
        data,targ = batch[0].unsqueeze(0),torch.tensor(batch[1]).unsqueeze(0)
        data_list.append(data)
        targ_list.append(targ)
    
    data_list = torch.cat(data_list,dim=0)
    targ_list = torch.cat(targ_list,dim=0)

    if args.cuda:
        data_list = data_list.cuda() 
        targ_list = targ_list.cuda()
        
    return data_list,targ_list 

def getExplanations(inds,data,predClassInds,attrFunc,kwargs,args):
    explanations = []
    for i,data_i,predClassInd in zip(inds,data,predClassInds):
        if args.att_metrics_post_hoc:
            explanation = applyPostHoc(attrFunc,data_i.unsqueeze(0),predClassInd,kwargs,args)
        else:
            explanation = attrFunc(i)
        explanations.append(explanation)
    explanations = torch.cat(explanations,dim=0)
    return explanations 

def add_validity_rate_multi_step(metric_name,all_score_list):
    if metric_name == "Deletion":
        validity_rate = (all_score_list[:,:-1] > all_score_list[:,1:]).astype("float").mean()
    else: 
        validity_rate = (all_score_list[:,:-1] < all_score_list[:,1:]).astype("float").mean()

    return validity_rate

def add_validity_rate_single_step(metric_name,all_score_list,all_score_masked_list):
    if metric_name == "IIC_AD":
        validity_rate = (all_score_list < all_score_masked_list).astype("float").mean()
    else: 
        validity_rate = (all_score_list > all_score_masked_list).astype("float").mean()
    return validity_rate 


def run_separability_analysis(repres1,repres2,normalize,seed,folds=10):

    len1 = len(repres1)
    len2 = len(repres2)

    if normalize:
        repres1 = repres1/np.abs(repres1).sum(axis=1,keepdims=True)
        repres2 = repres2/np.abs(repres2).sum(axis=1,keepdims=True)

    labels1 = np.zeros(len1).astype("int")
    labels2 = np.ones(len2).astype("int")

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_acc = []
    train_auc = []
    val_acc = []
    val_auc = []

    train_inv_auc = []
    val_inv_auc = []   

    for _ in range(folds):

        inds = np.random.permutation(len1)

        repr1_perm,lab1_perm = repres1[inds],labels1[inds]
        repr2_perm,lab2_perm = repres2[inds],labels2[inds]

        train_x = np.concatenate((repr1_perm[:len1//2],repr2_perm[:len2//2]),axis=0)
        test_x = np.concatenate((repr1_perm[len1//2:],repr2_perm[len2//2:]),axis=0)

        train_y = np.concatenate((lab1_perm[:len1//2],lab2_perm[:len2//2]),axis=0)
        test_y = np.concatenate((lab1_perm[len1//2:],lab2_perm[len2//2:]),axis=0)

        model = svm.SVC(probability=True)
        model.fit(train_x,train_y)

        train_y_score = model.predict_proba(train_x)[:,1]
        train_acc.append(model.score(train_x,train_y))
        train_auc.append(roc_auc_score(train_y,train_y_score))
        
        test_y_score = model.predict_proba(test_x)[:,1]
        val_acc.append(model.score(test_x,test_y))
        val_auc.append(roc_auc_score(test_y,test_y_score))

    train_acc,train_auc = np.array(train_acc),np.array(train_auc)
    val_acc,val_auc = np.array(val_acc),np.array(val_auc)

    train_inv_auc = np.array(train_inv_auc)
    val_inv_auc = np.array(val_inv_auc)

    return {"train_acc":train_acc,"train_auc":train_auc,"val_acc":val_acc,"val_auc":val_auc,"train_inv_auc":train_inv_auc,"val_inv_auc":val_inv_auc}

def interval_metric(a, b):
    return (a-b)**2

def krippendorff_alpha_bootstrap(*data,**kwargs):

    #print("krippendorff_alpha_bootstrap",len(data))
    #for i in range(len(data)):
    #    print("\t",data[i].shape)

    data = np.stack(data)
    
    if len(data.shape) == 3:
        data = data.transpose(1,0,2)
    else:
        data = data[np.newaxis]

    res_list = []

    #print("final shape",data.shape)
    for i in range(len(data)):
        res_list.append(krippendorff_alpha_paralel(data[i],**kwargs))
  
    return res_list

#From https://github.com/grrrr/krippendorff-alpha/blob/master/krippendorff_alpha.py
def krippendorff_alpha_paralel(data, metric=interval_metric, missing_items=None,axis=None):
    '''
    Calculate Krippendorff's alpha (inter-rater reliability):
    
    data is in the format
    [
        {unit1:value, unit2:value, ...},  # coder 1
        {unit1:value, unit3:value, ...},   # coder 2
        ...                            # more coders
    ]
    or 
    it is a sequence of (masked) sequences (list, numpy.array, numpy.ma.array, e.g.) with rows corresponding to coders and columns to items
    
    metric: function calculating the pairwise distance
    force_vecmath: force vector math for custom metrics (numpy required)
    convert_items: function for the type conversion of items (default: float)
    missing_items: indicator for missing items (default: None)
    '''
    
    # number of coders
    m = len(data)
    
    # set of constants identifying missing values
    if missing_items is None:
        maskitems = []
    else:
        maskitems = list(missing_items)
    if np is not None:
        maskitems.append(np.ma.masked_singleton)
    
    # convert input data to a dict of items
    units = {}

    #for d in data:
    #    diter = enumerate(d)
            
    #    for it, g in diter:
    #        try:
    #            its = units[it]
    #        except KeyError:
    #            its = []
    #            units[it] = its
    #        its.append(g)

    #print(data.shape)

    #print("krippendorff_alpha_paralel",data.shape)
    units = {j:data[:,j] for j in range(data.shape[1])}


    units = dict((it, d) for it, d in units.items() if len(d) > 1)  # units with pairable values
    #n = sum(len(pv) for pv in units.values())  # number of pairable values

    n = data.size
  
    Do = 0.

    data_perm = data.transpose(1,0)


    Do = metric(data_perm[:,:,np.newaxis],data_perm[:,np.newaxis,:]).sum()
  
    #for grades in units.values():
        
    #    Du = metric(grades[:,np.newaxis],grades[np.newaxis,:]).sum()

        #Du = sum(np.sum(metric(gr, gri)) for gri in gr)
        #Do += Du/float(len(grades)-1)
    #    Do += Du

    #Do /= n
    Do /= (n*(data.shape[0]-1))

    if Do == 0:
        return 1.

    De = metric(data_perm[np.newaxis,:,:,np.newaxis],data_perm[:,np.newaxis,np.newaxis,:]).sum()
  
    De /= float(n*(n-1))

    coeff = 1.-Do/De if (Do and De) else 1.

    #print(data.mean(),coeff,Do,De)

    return coeff

def krippendorff_alpha(data, metric=interval_metric, convert_items=float, missing_items=None,axis=None):
    '''
    Calculate Krippendorff's alpha (inter-rater reliability):
    
    data is in the format
    [
        {unit1:value, unit2:value, ...},  # coder 1
        {unit1:value, unit3:value, ...},   # coder 2
        ...                            # more coders
    ]
    or 
    it is a sequence of (masked) sequences (list, numpy.array, numpy.ma.array, e.g.) with rows corresponding to coders and columns to items
    
    metric: function calculating the pairwise distance
    force_vecmath: force vector math for custom metrics (numpy required)
    convert_items: function for the type conversion of items (default: float)
    missing_items: indicator for missing items (default: None)
    '''
    
    # number of coders
    m = len(data)
    
    # set of constants identifying missing values
    if missing_items is None:
        maskitems = []
    else:
        maskitems = list(missing_items)
    if np is not None:
        maskitems.append(np.ma.masked_singleton)
    
    # convert input data to a dict of items
    units = {}

    for d in data:
        try:
            # try if d behaves as a dict
            diter = d.items()
        except AttributeError:
            # sequence assumed for d
            diter = enumerate(d)
            
        for it, g in diter:
            if g not in maskitems:
                try:
                    its = units[it]
                except KeyError:
                    its = []
                    units[it] = its
                its.append(convert_items(g))

    units = dict((it, d) for it, d in units.items() if len(d) > 1)  # units with pairable values
    n = sum(len(pv) for pv in units.values())  # number of pairable values
    
    if n == 0:
        raise ValueError("No items to compare.")
        
    Do = 0.
    for grades in units.values():
        gr = np.asarray(grades)
        Du = sum(np.sum(metric(gr, gri)) for gri in gr)
        Do += Du/float(len(grades)-1)
    Do /= float(n)

    if Do == 0:
        return 1.

    De = 0.
    for g1 in units.values():
        d1 = np.asarray(g1)
        for g2 in units.values():
            De += sum(np.sum(metric(d1, gj)) for gj in g2)
    De /= float(n*(n-1))

    return 1.-Do/De if (Do and De) else 1.


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
