import os
import glob

import numpy as np
import torch
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from saliency_maps_metrics.multi_step_metrics import Deletion, Insertion
from saliency_maps_metrics.single_step_metrics import IIC_AD, ADD
from saliency_maps_metrics.multi_step_metrics import compute_auc_metric
from captum.attr import (IntegratedGradients,NoiseTunnel,LayerGradCam,GuidedBackprop)

import args
from args import ArgReader
from args import str2bool
import modelBuilder
import load_data
import metrics
from trainVal import addInitArgs,preprocessAndLoadParams
from post_hoc_expl.gradcam import GradCAMpp
from post_hoc_expl.scorecam import ScoreCam
from post_hoc_expl.xgradcam import AblationCAM,XGradCAM
from rise import RISE

from utils import normalize_tensor

is_multi_step_dic = {"Deletion":True,"Insertion":True,"IIC_AD":False,"ADD":False}
const_dic = {"Deletion":Deletion,"Insertion":Insertion,"IIC_AD":IIC_AD,"ADD":ADD}

def get_metric_dics():
    return is_multi_step_dic,const_dic

def find_other_class_labels(inds,testDataset):

    inds_bckgr = (inds + len(testDataset)//2) % len(testDataset)

    labels = np.array([testDataset[ind][1] for ind in inds])
    labels_bckgr = np.array([testDataset[ind][1] for ind in inds_bckgr])

    while (labels==labels_bckgr).any():
        for i in range(len(inds)):
            if inds[i] == inds_bckgr[i]:
                inds_bckgr[i] = torch.randint(len(testDataset),size=(1,))[0]

        inds_bckgr = torch.randint(len(testDataset),size=(len(inds),))
        labels_bckgr = np.array([testDataset[ind][1] for ind in inds_bckgr])

    return inds_bckgr

#From https://kai760.medium.com/how-to-use-torch-fft-to-apply-a-high-pass-filter-to-an-image-61d01c752388
def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) 
            if i != axis else slice(0, n, None) 
            for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) 
            if i != axis else slice(n, None, None) 
            for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)
    
def fftshift(X):
    real, imag = X.chunk(chunks=2, dim=-1)
    real, imag = real.squeeze(dim=-1), imag.squeeze(dim=-1)    
    for dim in range(2, len(real.size())):
        real = roll_n(real, axis=dim,n=int(np.ceil(real.size(dim) / 2)))
        imag = roll_n(imag, axis=dim,n=int(np.ceil(imag.size(dim) / 2)))
    real, imag = real.unsqueeze(dim=-1), imag.unsqueeze(dim=-1)
    X = torch.cat((real,imag),dim=1)
    return torch.squeeze(X)
    
def ifftshift(X):
    real, imag = X.chunk(chunks=2, dim=-1)
    real, imag = real.squeeze(dim=-1), imag.squeeze(dim=-1)
    for dim in range(len(real.size()) - 1, 1, -1):
        real = roll_n(real, axis=dim,n=int(np.floor(real.size(dim) / 2)))
        imag = roll_n(imag, axis=dim,n=int(np.floor(imag.size(dim) / 2)))
    real, imag = real.unsqueeze(dim=-1), imag.unsqueeze(dim=-1)
    X = torch.cat((real, imag), dim=1)
    return torch.squeeze(X)

def maskData(data,attMaps_interp,data_bckgr):
    data_masked = data*attMaps_interp+data_bckgr*(1-attMaps_interp)
    return data_masked 

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
            print(data_i.shape,data.shape)
            explanation = applyPostHoc(attrFunc,data_i.unsqueeze(0),predClassInd,kwargs,args)
        else:
            explanation = attrFunc(i)
        explanations.append(explanation)
    explanations = torch.cat(explanations,dim=0)
    return explanations 

def computeSpars(data_shape,attMaps,args,resDic):
    if args.att_metrics_post_hoc:
        features = None 
    else:
        features = resDic["feat"]
        if "attMaps" in resDic:
            attMaps = resDic["attMaps"]
        else:
            attMaps = torch.ones(data_shape[0],1,features.size(2),features.size(3)).to(features.device)

    sparsity = metrics.compAttMapSparsity(attMaps,features)
    sparsity = sparsity/data_shape[0]
    return sparsity 

def inference(net,data,predClassInd,args):
    if not args.prototree:
        resDic = net(data)
        score = torch.softmax(resDic["pred"],dim=-1)[:,predClassInd[0]]
        feat = resDic["feat_pooled"]
    else:
        score = net(data)[0][:,predClassInd[0]]
        feat = None
    return score,feat

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
    
def getAttMetrMod(net,testDataset,args):
    if args.att_metrics_post_hoc == "gradcam":
        netGradMod = modelBuilder.GradCamMod(net)
        attrMod = LayerGradCam(netGradMod.forward,netGradMod.layer4)
        attrFunc = attrMod.attribute
        kwargs = {}
    elif args.att_metrics_post_hoc == "gradcam_pp":
        netGradMod = modelBuilder.GradCamMod(net)
        model_dict = dict(type=args.first_mod, arch=netGradMod, layer_name='layer4', input_size=(448, 448))
        attrMod = GradCAMpp(model_dict,True)
        attrFunc = attrMod.__call__
        kwargs = {}
    elif args.att_metrics_post_hoc == "guided":
        netGradMod = modelBuilder.GradCamMod(net)
        attrMod = GuidedBackprop(netGradMod)
        attrFunc = attrMod.attribute
        kwargs = {}
    elif args.att_metrics_post_hoc == "xgradcam":
        netGradMod = modelBuilder.GradCamMod(net)
        attrMod = XGradCAM(model=netGradMod,target_layers=netGradMod.layer4,use_cuda=args.cuda)
        attrFunc = attrMod
        kwargs = {}
    elif args.att_metrics_post_hoc == "ablation_cam":
        netGradMod = modelBuilder.GradCamMod(net)
        attrMod = AblationCAM(model=netGradMod,target_layers=netGradMod.layer4,use_cuda=args.cuda)
        attrFunc = attrMod
        kwargs = {}
    elif args.att_metrics_post_hoc == "score_cam":
        attrMod = ScoreCam(net)
        attrFunc = attrMod.generate_cam
        kwargs = {}
    elif args.att_metrics_post_hoc == "varGrad" or args.att_metrics_post_hoc == "smoothGrad":
        torch.backends.cudnn.benchmark = False
        torch.set_grad_enabled(False)
        
        net.eval()
        netGradMod = modelBuilder.GradCamMod(net)

        ig = IntegratedGradients(netGradMod)
        attrMod = NoiseTunnel(ig)
        attrFunc = attrMod.attribute

        batch = testDataset.__getitem__(0)
        data_base = torch.zeros_like(batch[0].unsqueeze(0))

        if args.cuda:
            data_base = data_base.cuda()
        kwargs = {"nt_type":'smoothgrad_sq' if args.att_metrics_post_hoc == "smoothGrad" else "vargrad", \
                        "stdevs":0.02, "nt_samples":3,"nt_samples_batch_size":3}
    elif args.att_metrics_post_hoc == "rise":
        torch.set_grad_enabled(False)
        attrMod = RISE(net)
        attrFunc = attrMod.__call__
        kwargs = {}
    else:
        raise ValueError("Unknown post-hoc method",args.att_metrics_post_hoc)
    return attrFunc,kwargs

def loadSalMaps(exp_id,model_id):

    norm_paths = glob.glob(f"../results/{exp_id}/norm_{model_id}_epoch*.npy")
    if len(norm_paths) != 1:
        raise ValueError(f"Wrong norm path number for exp {exp_id} model {model_id}")
    else:
        norm = torch.tensor(np.load(norm_paths[0],mmap_mode="r"))/255.

    attMaps_paths = glob.glob(norm_paths[0].replace("norm","attMaps"))
    if len(attMaps_paths) >1:
        raise ValueError(f"Wrong path number for exp {exp_id} model {model_id}")
    elif len(attMaps_paths) == 0:
        print(f"No attMaps found for {model_id}. Only using the norm.")
        salMaps = norm
    else:
        attMaps = torch.tensor(np.load(attMaps_paths[0],mmap_mode="r"))/255.0
        salMaps = attMaps*norm
        salMaps = normalize_tensor(salMaps,dim=(1,2,3))
        
    return salMaps

def init_post_hoc_arg(argreader):
    argreader.parser.add_argument('--att_metrics_post_hoc', type=str, help='The post-hoc method to use instead of the model ')
    return argreader

def main(argv=None):
    # Getting arguments from config file and command line
    # Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--attention_metric', type=str, help='The attention metric to compute.')
    argreader.parser.add_argument('--att_metrics_img_nb', type=int, help='The nb of images on which to compute the att metric.')
    
    argreader.parser.add_argument('--att_metrics_map_resolution', type=int, help='The resolution at which to resize the attention map.\
                                    If this value is None, the map will not be resized.')
    argreader.parser.add_argument('--att_metrics_sparsity_factor', type=float, help='Used to increase (>1) or decrease (<1) the sparsity of the saliency maps.\
                                    Set to None to not modify the sparsity.')

    argreader.parser.add_argument('--att_metrics_max_brnpa', type=str2bool, help='To agregate br-npa maps with max instead of mean')
    argreader.parser.add_argument('--att_metrics_onlyfirst_brnpa', type=str2bool, help='To agregate br-npa maps with max instead of mean')
    argreader.parser.add_argument('--att_metrics_few_steps', type=str2bool, help='To do as much step for high res than for low res')
    argreader.parser.add_argument('--att_metr_do_again', type=str2bool, help='To run computation if already done',default=True)

    argreader.parser.add_argument('--data_replace_method', type=str, help='The pixel replacement method.',default="black")
        
    argreader.parser.add_argument('--att_metr_save_feat', type=str2bool, help='',default=False)
    argreader.parser.add_argument('--att_metr_add_first_feat', type=str2bool, help='',default=False)

    argreader = addInitArgs(argreader)
    argreader = init_post_hoc_arg(argreader)
    argreader = modelBuilder.addArgs(argreader)
    argreader = load_data.addArgs(argreader)

    # Reading the comand line arg
    argreader.getRemainingArgs()

    args = argreader.args
  
    result_file_path = f"../results/{args.exp_id}/{args.attention_metric}_{args.model_id}.npy"

    if args.att_metr_do_again or (not os.path.exists(result_file_path)):

        args.val_batch_size = 1
        _,testDataset = load_data.buildTestLoader(args, "test")

        bestPath = glob.glob(f"../models/{args.exp_id}/model{args.model_id}_best_epoch*")[0]

        net = modelBuilder.netBuilder(args)
        net = preprocessAndLoadParams(bestPath,args.cuda,net)
        net.eval()
        net_lambda = lambda x:torch.softmax(net(x)["pred"],dim=-1)
        
        if args.att_metrics_post_hoc:
            attrFunc,kwargs = getAttMetrMod(net,testDataset,args)
        else:
            salMaps_dataset = loadSalMaps(args.exp_id,args.model_id)
            attrFunc = lambda i:(salMaps_dataset[i,0:1]).unsqueeze(0)
            kwargs = None

        if args.att_metrics_post_hoc != "gradcam_pp":
            torch.set_grad_enabled(False)

        nbImgs = args.att_metrics_img_nb
        torch.manual_seed(0)
        inds = torch.randint(len(testDataset),size=(nbImgs,))

        data,_ = getBatch(testDataset,inds,args)

        predClassInds = net_lambda(data).argmax(dim=-1)
        
        explanations = getExplanations(inds,data,predClassInds,attrFunc,kwargs,args)

        is_multi_step_dic,const_dic = get_metric_dics()

        if args.data_replace_method is None:
            arg_list = []
        else:
            arg_list = [args.data_replace_method]

        if is_multi_step_dic[args.attention_metric]:
            metric = const_dic[args.attention_metric](*arg_list)
            scores,saliency_scores = metric.compute_scores(net_lambda,data,explanations,predClassInds)
            saved_dic = {"prediction_scores":scores,"saliency_scores":saliency_scores}

        else:
            metric = const_dic[args.attention_metric](*arg_list)
            scores,scores_masked = metric.compute_scores(net_lambda,data,explanations,predClassInds)
            saved_dic = {"prediction_scores":scores,"prediction_scores_with_mask":scores_masked}
        
        np.save(result_file_path,saved_dic)

if __name__ == "__main__":
    main()