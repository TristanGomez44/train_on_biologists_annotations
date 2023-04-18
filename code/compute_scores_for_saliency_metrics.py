import os,sys
import glob

import numpy as np
import torch
import matplotlib.pyplot as plt
plt.switch_backend('agg')
from captum.attr import (IntegratedGradients,NoiseTunnel,LayerGradCam,GuidedBackprop)

import hashlib

from args import ArgReader,str2bool,addInitArgs,init_post_hoc_arg,addLossTermArgs
import modelBuilder
import load_data
from init_model import preprocessAndLoadParams
from post_hoc_expl.scorecam import ScoreCam
from post_hoc_expl.xgradcam import AblationCAM,XGradCAM,AblationCAM_NoUpScale
from post_hoc_expl.rise import RISE
from post_hoc_expl.gradcampp import LayerGradCampp
from post_hoc_expl.baselines import AM,CAM,RandomMap,TopFeatMap,RandomFeatMap
from post_hoc_expl.ablationcam2 import AblationCAM2
from metrics import sample_img_inds,get_sal_metric_dics,getBatch,getExplanations
from utils import normalize_tensor

def find_class_first_image_inds(label_list):
    class_first_image_inds = [0]
    labels = [0]
    for i in range(len(label_list)):
        label = label_list[i]
        if label not in labels:
            labels.append(label)
            class_first_image_inds.append(i)
    
    return class_first_image_inds

def get_other_img_inds(inds):
    other_img_inds = np.zeros_like(inds)
    shift = (np.arange(len(inds)) + len(inds)//2) % len(inds)
    other_img_inds = inds[shift]
    return other_img_inds

def getAttMetrMod(net,testDataset,args):

    baseline_dict = {"am":AM,"cam":CAM,"randommap":RandomMap,"topfeatmap":TopFeatMap,"randomfeatmap":RandomFeatMap}

    if args.att_metrics_post_hoc in baseline_dict.keys():
        attrMod = baseline_dict[args.att_metrics_post_hoc](net)
        attrFunc = attrMod.forward
        kwargs = {}    
    elif args.att_metrics_post_hoc == "cam":
        attrMod = CAM(net)
        attrFunc = attrMod.forward
        kwargs = {}
    elif args.att_metrics_post_hoc == "gradcam":
        netGradMod = modelBuilder.GradCamMod(net)
        attrMod = LayerGradCam(netGradMod.forward,netGradMod.layer4)
        attrFunc = attrMod.attribute
        kwargs = {}
    elif args.att_metrics_post_hoc == "gradcampp":
        netGradMod = modelBuilder.GradCamMod(net)
        attrMod = LayerGradCampp(netGradMod.forward,netGradMod.layer4)
        attrFunc = attrMod.attribute
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
    elif args.att_metrics_post_hoc == "ablationcam":
        netGradMod = modelBuilder.GradCamMod(net)
        attrMod = AblationCAM(model=netGradMod,target_layers=netGradMod.layer4,use_cuda=args.cuda,batch_size=args.ablationcam_batchsize)
        attrFunc = attrMod
        kwargs = {}
    elif args.att_metrics_post_hoc == "ablationcam2":
        netGradMod = modelBuilder.GradCamMod(net)
        attrMod = AblationCAM2(model=netGradMod,target_layers=netGradMod.layer4,use_cuda=args.cuda,batch_size=args.ablationcam_batchsize)
        attrFunc = attrMod
        kwargs = {}
    elif args.att_metrics_post_hoc == "ablationcamnous":
        netGradMod = modelBuilder.GradCamMod(net)
        attrMod = AblationCAM_NoUpScale(model=netGradMod,target_layers=netGradMod.layer4,use_cuda=args.cuda,batch_size=args.ablationcam_batchsize)
        attrFunc = attrMod
        kwargs = {}
    elif args.att_metrics_post_hoc == "scorecam":
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

    if not "transf" in model_id:
        norm_paths = glob.glob(f"../results/{exp_id}/norm_{model_id}_epoch*.npy")
   
        if len(norm_paths) != 1:
            raise ValueError(f"Wrong norm path number for exp {exp_id} model {model_id}")
        else:
            norm = torch.tensor(np.load(norm_paths[0],mmap_mode="r"))/255.
    else:
        norm_paths = []
        norm = torch.ones((1,1,1,1))
    
    if len(norm_paths) == 0:
        attMaps_paths = glob.glob(f"../results/{exp_id}/attMaps_{model_id}_epoch*.npy")
    else:
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

def get_attr_func(net,testDataset,args):
    if args.att_metrics_post_hoc:
        attrFunc,kwargs = getAttMetrMod(net,testDataset,args)
    else:
        salMaps_dataset = loadSalMaps(args.exp_id,args.model_id)

        def att_maps_attr_func(inds):
            if not type(inds) is list:
                inds = [inds]  
            maps = salMaps_dataset[inds,0:1]
            return maps

        attrFunc = att_maps_attr_func
        kwargs = {}
    return attrFunc,kwargs

def compute_or_load_explanations(inds,args,data,predClassInds,attrFunc,kwargs):
    inds_string = "-".join([str(ind) for ind in inds])
    hashed_inds = hashlib.sha1(inds_string.encode("utf-8")).hexdigest()[:16]
    torch.save(inds,f"../results/{args.exp_id}/inds_{hashed_inds}.th")

    expl_path = f"../results/{args.exp_id}/explanations_{args.model_id}_{args.att_metrics_post_hoc}_{hashed_inds}.th"
    if not os.path.exists(expl_path):
        print("Computing explanations")
        explanations = getExplanations(inds,data,predClassInds,attrFunc,kwargs,args)
        torch.save(explanations.cpu(),expl_path)
    else:
        print("Already computed explanations")
        explanations = torch.load(expl_path).to(data.device)
    return explanations

def main(argv=None):
    # Getting arguments from config file and command line
    # Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--attention_metric', type=str, help='The attention metric to compute.')
    argreader.parser.add_argument('--do_again', type=str2bool, help='To run computation if already done')
    argreader.parser.add_argument('--data_replace_method', type=str, help='The pixel replacement method.')
    argreader.parser.add_argument('--cumulative', type=str2bool, help='To prevent acumulation of perturbation when computing metrics.',default=True)

    argreader = addInitArgs(argreader)
    argreader = init_post_hoc_arg(argreader)
    argreader = modelBuilder.addArgs(argreader)
    argreader = load_data.addArgs(argreader)
    argreader = addLossTermArgs(argreader)

    # Reading the comand line arg
    argreader.getRemainingArgs()

    args = argreader.args

    #Constructing result file path
    post_hoc_suff = "" if args.att_metrics_post_hoc is None else "-"+args.att_metrics_post_hoc
    formated_attention_metric = args.attention_metric.replace("_","")

    #Constructing metric
    is_multi_step_dic,const_dic = get_sal_metric_dics()
    if args.data_replace_method=="default" or args.data_replace_method is None or args.data_replace_method == "otherimage":
        metric_constr_arg_dict = {}
    else:
        metric_constr_arg_dict = {"data_replace_method":args.data_replace_method}

    if is_multi_step_dic[args.attention_metric]:
        metric_constr_arg_dict.update({"cumulative":args.cumulative})

        if not args.cumulative:
            formated_attention_metric += "nc"

    metric = const_dic[args.attention_metric](**metric_constr_arg_dict)

    if args.data_replace_method is None or args.data_replace_method == "default":
        data_replace_method = metric.data_replace_method 
    else:
        data_replace_method = args.data_replace_method
        
    result_file_path = f"../results/{args.exp_id}/{formated_attention_metric}-{data_replace_method}_{args.model_id}{post_hoc_suff}.npy"
    
    print(result_file_path)
    result_file_exists = os.path.exists(result_file_path)
    result_file_misses_supp_keys = result_file_exists and len(np.load(result_file_path,allow_pickle=True).item().keys()) < 5

    if args.do_again or (not result_file_exists) or result_file_misses_supp_keys:
  
        _,testDataset = load_data.buildTestLoader(args, "test")

        bestPath = glob.glob(f"../models/{args.exp_id}/model{args.model_id}_best_epoch*")[0]

        net = modelBuilder.netBuilder(args)
        net = preprocessAndLoadParams(bestPath,args.cuda,net,verbose=False)
        net.eval()
        net_lambda = lambda x:net(x)["output"]
        
        attrFunc,kwargs = get_attr_func(net,testDataset,args)

        inds = sample_img_inds(args.img_nb_per_class,testDataset=testDataset)

        data,target = getBatch(testDataset,inds,args)
        
        if args.data_replace_method == "otherimage":
            other_img_inds = get_other_img_inds(inds)
            other_data,_ = getBatch(testDataset,other_img_inds,args)
        else:
            other_img_inds = None

        torch.set_grad_enabled(False)
        outputs =  net_lambda(data)
    
        predClassInds = outputs.argmax(dim=-1)
        
        if args.att_metrics_post_hoc == "gradcam_pp":
            torch.set_grad_enabled(True)

        if result_file_exists and result_file_misses_supp_keys:
            print("Just adding missing supplementary keys...")
            result_dic = np.load(result_file_path,allow_pickle=True).item()
            result_dic.update({"outputs":outputs.cpu(),"target":target.cpu(),"inds":inds.cpu()})
            np.save(result_file_path,result_dic)
        else:
            explanations = compute_or_load_explanations(inds,args,data,predClassInds,attrFunc,kwargs)
            print(explanations.shape,data.shape)

            torch.set_grad_enabled(False)   
    
            if args.debug:
                data = data[:2]
                if args.data_replace_method == "otherimage" and other_data is not None:
                    other_data = other_data[:2]

            metric_args = [net_lambda,data,explanations,predClassInds]
            kwargs = {"save_all_class_scores":True}

            if args.data_replace_method == "otherimage":
                metric_args.append(other_data)

            if is_multi_step_dic[args.attention_metric]:  
                scores,saliency_scores = metric.compute_scores(*metric_args,**kwargs)
                saved_dic = {"prediction_scores":scores,"saliency_scores":saliency_scores,"outputs":outputs.cpu(),"target":target.cpu(),"inds":inds.cpu()}
            else:
                scores,scores_masked = metric.compute_scores(*metric_args,**kwargs)
                saved_dic = {"prediction_scores":scores,"prediction_scores_with_mask":scores_masked,"outputs":outputs.cpu(),"target":target.cpu(),"inds":inds.cpu()}
            
            np.save(result_file_path,saved_dic)
    else:
        print("Already done")
        
if __name__ == "__main__":
    main()