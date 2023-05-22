from re import I
import sys
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import math
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn import svm
 
from utils import remove_no_annot

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

def get_ylim(metric):
    if metric in ["DC","IC","ADD"]:
        return (-1,1)
    else:
        return (0,1)

def get_correlation_metric_list():
    return ["DC","IC"]

def get_sub_multi_step_metric_list():
    return ["DAUC","DC","IAUC","IC"]

def get_sub_single_step_metric_list():
    return ["AD","IIC","ADD"]

def get_metrics_to_minimize():
    return ["DAUC","AD"]

def get_sub_metric_list():
    return get_sub_multi_step_metric_list() + get_sub_single_step_metric_list()

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

def compute_metrics(target_dic,resDict):

    metDict = {}
    for key in target_dic.keys():
        output,target = remove_no_annot(resDict["output_"+key],target_dic[key])
        metDict["Accuracy_{}".format(key)] = compAccuracy(output,target)

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

    if "attMaps" in resDict:
        explanations = resDict["attMaps"]
    else:
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

#From https://github.com/torrvision/focal_calibration/blob/main/Metrics/metrics.py
def expected_calibration_error(var_dic,metrDict):

    func_dic= {"ECE":ece,"AdaECE":ada_ece,"ClassECE":class_ece}

    for metric_name in func_dic:

        metrDict[metric_name] = func_dic[metric_name](var_dic["output"], var_dic["target"])

        if "output_masked" in var_dic:
            metrDict[metric_name+"_masked"] = func_dic[metric_name](var_dic["output_masked"], var_dic["target"])
    
    return metrDict

def ece(logits, labels,n_bins=15,return_conf_and_acc=False):
    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
   
    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)

    ece = torch.zeros(1, device=logits.device)
    conf_list,acc_list = [],[]
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

            acc_list.append(accuracy_in_bin)
        else:
            acc_list.append(0)

    if return_conf_and_acc:
        return ece.item(),bin_boundaries,acc_list
    else:
        return ece.item()

def histedges_equalN(x,nbins):
    npt = len(x)
    return np.interp(np.linspace(0, npt, nbins + 1),
                    np.arange(npt),
                    np.sort(x))

def ada_ece(logits, labels,n_bins=15):

    softmaxes = F.softmax(logits, dim=1)
    confidences, predictions = torch.max(softmaxes, 1)
    accuracies = predictions.eq(labels)
    n, bin_boundaries = np.histogram(confidences.cpu().detach(), histedges_equalN(confidences.cpu().detach(),n_bins))
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    ece = torch.zeros(1, device=logits.device)
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Calculated |confidence - accuracy| in each bin
        in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
        prop_in_bin = in_bin.float().mean()
        if prop_in_bin.item() > 0:
            accuracy_in_bin = accuracies[in_bin].float().mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece.item()

def class_ece(logits, labels,n_bins=15):

    bin_boundaries = torch.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    num_classes = int((torch.max(labels) + 1).item())
    softmaxes = F.softmax(logits, dim=1)
    per_class_sce = None

    for i in range(num_classes):
        class_confidences = softmaxes[:, i]
        class_sce = torch.zeros(1, device=logits.device)
        labels_in_class = labels.eq(i) # one-hot vector of all positions where the label belongs to the class i

        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = class_confidences.gt(bin_lower.item()) * class_confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = labels_in_class[in_bin].float().mean()
                avg_confidence_in_bin = class_confidences[in_bin].mean()
                class_sce += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        if (i == 0):
            per_class_sce = class_sce
        else:
            per_class_sce = torch.cat((per_class_sce, class_sce), dim=0)

    sce = torch.mean(per_class_sce)
    return sce.item()

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

    bs = args.val_batch_size
    batch_nb = len(data)//bs + 1*(len(data)%args.val_batch_size>0)
    all_expl = []
    for i in range(batch_nb):
        ind,data_i,predClassInd = inds[i*bs:(i+1)*bs],data[i*bs:(i+1)*bs],predClassInds[i*bs:(i+1)*bs]
        if args.att_metrics_post_hoc:
            explanations = applyPostHoc(attrFunc,data_i,predClassInd,kwargs,args)
        else:
            explanations = attrFunc(ind)   
        all_expl.append(explanations)
    all_expl = torch.cat(all_expl,dim=0)

    return all_expl 

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

def ratio_metric(a,b):

    if a.dtype == "bool":
        result = (a==b)*1.0
    else:
        result = (((a-b)/(a+b))**2)
        result[a+b == 0] = 0

    return result

def krippendorff_alpha_bootstrap(*data,**kwargs):

    data = np.stack(data)
    
    if len(data.shape) == 3:
        data = data.transpose(1,0,2)
    else:
        data = data[np.newaxis]

    res_list = []

    for i in range(len(data)):
        res_list.append(krippendorff_alpha_paralel(data[i],**kwargs))
  
    return res_list

def make_n_dict(data):

    unit_nb = data.shape[1]

    o = np.zeros((unit_nb,unit_nb))

    for i in range(unit_nb):
        for j in range(unit_nb):
            o[i,j] = 0
            for u in range(unit_nb):
                number_of_ij_pairs = (data[:,u] == i+1).sum()*(data[:,u] == j+1).sum()
                o[i,j] += number_of_ij_pairs/(unit_nb-1)

    n_vector = o.sum(axis=1)

    diff_mat = np.zeros((unit_nb,unit_nb))

    for i in range(unit_nb):
        for j in range(unit_nb):
            if i >= j:
                start,end = j,i 
            else:
                start,end = i,j

            diff_mat[i,j] = sum([n_vector[k] for k in range(start,end+1)])
            diff_mat[i,j] -= (n_vector[i] + n_vector[j])/2

    diff_mat = np.power(diff_mat,2)

    return diff_mat

def ordinal_metric(a,b,diff_mat):

    shape = np.broadcast(a,b).shape

    a = np.broadcast_to(a,shape)
    b = np.broadcast_to(b,shape)

    diff = diff_mat[a.reshape(-1)-1,b.reshape(-1)-1]

    diff = diff.reshape(shape)

    return diff

def binary_metric(a,b):
    return (a==b)*1.0

metric_dict = {"ratio_metric":ratio_metric,"interval_metric":interval_metric,"ordinal_metric":ordinal_metric,"binary_metric":binary_metric}


#From https://github.com/grrrr/krippendorff-alpha/blob/master/krippendorff_alpha.py
def krippendorff_alpha_paralel(data, metric=ratio_metric, missing_items=None,axis=None):
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

    if data.dtype == np.int64:
        metric= "ordinal_metric"
    elif data.dtype == bool:
        metric = "binary_metric"
        data = data.astype("int")

    if type(metric) is str:
        metric = metric_dict[metric]

    if metric is ordinal_metric:
        
        diff_mat=make_n_dict(data)

        metric = lambda a,b:ordinal_metric(a,b,diff_mat)


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
    units = {j:data[:,j] for j in range(data.shape[1])}
    units = dict((it, d) for it, d in units.items() if len(d) > 1)  # units with pairable values

    n = data.size
  
    Do = 0.

    data_perm = data.transpose(1,0)

    Do = metric(data_perm[:,:,np.newaxis],data_perm[:,np.newaxis,:]).sum()  
    Do /= (n*(data.shape[0]-1))

    if Do == 0:
        return 1.

    De = metric(data_perm[np.newaxis,:,:,np.newaxis],data_perm[:,np.newaxis,np.newaxis,:]).sum()
  
    De /= float(n*(n-1))

    coeff = 1.-Do/De if (Do and De) else 1.

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

    p_list=np.arange(11)/10

    for p in p_list:
        data = np.arange(4)[np.newaxis]

        data = data.repeat(int(p*100),0)
 
        data_rand = np.zeros((int((1-p)*100),4))
        for i in range(len(data_rand)):
            data_rand[i] = np.random.permutation(4)+1
        
        data = np.concatenate((data,data_rand),axis=0)

        data = data.astype("int")
 
        alpha = krippendorff_alpha_paralel(data, metric=ordinal_metric)

        print(p,alpha)

if __name__ == "__main__":

    main()
