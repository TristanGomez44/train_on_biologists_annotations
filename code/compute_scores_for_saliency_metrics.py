import os
import glob

import numpy as np
from numpy.random import Generator, PCG64
import torch
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from saliency_maps_metrics.multi_step_metrics import Deletion, Insertion
from saliency_maps_metrics.single_step_metrics import IIC_AD, ADD
from captum.attr import (IntegratedGradients,NoiseTunnel,LayerGradCam,GuidedBackprop)

from args import ArgReader
from args import str2bool
import modelBuilder
import load_data
from trainVal import addInitArgs,preprocessAndLoadParams
from post_hoc_expl.gradcam import GradCAMpp
from post_hoc_expl.scorecam import ScoreCam
from post_hoc_expl.xgradcam import AblationCAM,XGradCAM
from post_hoc_expl.rise import RISE

from utils import normalize_tensor

is_multi_step_dic = {"Deletion":True,"Insertion":True,"IIC_AD":False,"ADD":False}
const_dic = {"Deletion":Deletion,"Insertion":Insertion,"IIC_AD":IIC_AD,"ADD":ADD}

def get_metric_dics():
    return is_multi_step_dic,const_dic

def find_class_first_image_inds(testDataset):
    class_first_image_inds = [0]
    labels = [0]
    for i in range(len(testDataset)):
        label = testDataset.image_label[i]
        if label not in labels:
            labels.append(label)
            class_first_image_inds.append(i)
    
    return class_first_image_inds

def sample_img_inds(testDataset,nb_per_class):

    class_first_image_inds = find_class_first_image_inds(testDataset)

    nb_classes_to_be_sampled = len(class_first_image_inds)

    rng = Generator(PCG64())
    all_chosen_inds = []
    for label in range(nb_classes_to_be_sampled):

        startInd = class_first_image_inds[label]
        endInd = class_first_image_inds[label+1] if label+1<len(class_first_image_inds) else len(testDataset)
        candidate_inds = np.arange(startInd,endInd)

        if len(candidate_inds) < nb_per_class:
            raise ValueError(f"Number of image to be sampled per class is too high for class {label} which has only {len(candidate_inds)} images.")
        
        chosen_inds = rng.choice(candidate_inds, size=(nb_per_class,),replace=False)
        all_chosen_inds.append(chosen_inds)
    
    all_chosen_inds = np.concatenate(all_chosen_inds,axis=0)

    return all_chosen_inds 

def get_other_img_inds(inds):
    other_img_inds = np.zeros_like(inds)
    shift = (np.arange(len(inds)) + len(inds)//2) % len(inds)
    other_img_inds = inds[shift]
    return other_img_inds

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
    argreader.parser.add_argument('--img_nb_per_class', type=int, help='The nb of images on which to compute the att metric.')    
    argreader.parser.add_argument('--do_again', type=str2bool, help='To run computation if already done',default=True)
    argreader.parser.add_argument('--data_replace_method', type=str, help='The pixel replacement method.',default="black")

    argreader = addInitArgs(argreader)
    argreader = init_post_hoc_arg(argreader)
    argreader = modelBuilder.addArgs(argreader)
    argreader = load_data.addArgs(argreader)

    # Reading the comand line arg
    argreader.getRemainingArgs()

    args = argreader.args
  
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

    np.random.seed(0)
    inds = sample_img_inds(testDataset,args.img_nb_per_class)

    data,_ = getBatch(testDataset,inds,args)
    
    if args.data_replace_method == "otherimage":
        other_img_inds = get_other_img_inds(inds)
        other_data,_ = getBatch(testDataset,other_img_inds,args)
    else:
        other_img_inds = None

    predClassInds = net_lambda(data).argmax(dim=-1)
    explanations = getExplanations(inds,data,predClassInds,attrFunc,kwargs,args)
    is_multi_step_dic,const_dic = get_metric_dics()

    if args.data_replace_method is None or args.data_replace_method == "otherimage":
        metric_constr_arg_list = []
    else:
        metric_constr_arg_list = [args.data_replace_method]

    metric = const_dic[args.attention_metric](*metric_constr_arg_list)
    data_replace_method = metric.data_replace_method if args.data_replace_method is None else args.data_replace_method
    
    result_file_path = f"../results/{args.exp_id}/{args.attention_metric}-{data_replace_method}_{args.model_id}.npy"
    
    if args.debug:
        data = data[:2]
        if other_data is not None:
            other_data = other_data[:2]

    metric_args = [net_lambda,data,explanations,predClassInds]
    if args.data_replace_method == "otherimage":
        metric_args.append(other_data)

    if args.do_again or not os.path.exists(result_file_path):
        if is_multi_step_dic[args.attention_metric]:  
            scores,saliency_scores = metric.compute_scores(*metric_args)
            saved_dic = {"prediction_scores":scores,"saliency_scores":saliency_scores}
        else:
            scores,scores_masked = metric.compute_scores(*metric_args)
            saved_dic = {"prediction_scores":scores,"prediction_scores_with_mask":scores_masked}
        
        np.save(result_file_path,saved_dic)

if __name__ == "__main__":
    main()