import os
import glob
import math 

import matplotlib.pyplot as plt 
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import bootstrap

from args import ArgReader,init_post_hoc_arg,addLossTermArgs
import modelBuilder
import load_data
from init_model import preprocessAndLoadParams
from compute_scores_for_saliency_metrics import get_attr_func,compute_or_load_explanations
from metrics import get_sal_metric_dics,sample_img_inds,getBatch
from saliency_maps_metrics.multi_step_metrics import compute_auc_metric
from compute_saliency_metrics import apply_softmax
from does_resolution_impact_faithfulness import get_data_inds_and_explanations,load_model
import utils 

def blur_data(data,kernel_size):
    kernel = torch.ones(kernel_size,kernel_size)
    kernel = kernel/kernel.numel()
    kernel = kernel.unsqueeze(0).unsqueeze(0).expand(3,1,-1,-1)
    kernel = kernel.to(data.device)
    data = F.conv2d(data,kernel,padding=kernel.size(-1)//2,groups=kernel.size(0))  
    return data

def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)
    argreader = load_data.addArgs(argreader)
    argreader = modelBuilder.addArgs(argreader)
    argreader = init_post_hoc_arg(argreader)
    argreader = addLossTermArgs(argreader)

    #Reading the comand line arg
    argreader.getRemainingArgs()

    #Getting the args from command line and config file
    args = argreader.args

    _,testDataset = load_data.buildTestLoader(args, "test")

    net,net_lambda = load_model(args)

    data,explanations,predClassInds,outputs,inds = get_data_inds_and_explanations(net,net_lambda,testDataset,args)
    batch_inds = torch.arange(len(data))

    if "ablationcam" in args.att_metrics_post_hoc:
        #Reloading model to remove hook put by ablationcam module
        net,net_lambda = load_model(args)

    powers = np.arange(math.log(128,2)+1)
    kernel_sizes = np.power(2,powers).astype(int)+1
    outputs = torch.softmax(outputs,dim=-1)[batch_inds,predClassInds]

    all_outputs = []

    for i,kernel_size in enumerate(kernel_sizes):
        data_blurred = blur_data(data,kernel_size)
        outputs_blurred = net_lambda(data_blurred)
        outputs_blurred = torch.softmax(outputs_blurred,dim=-1)[batch_inds,predClassInds]

        all_outputs.append(outputs_blurred)
        
    rng = np.random.default_rng(0)

    all_outputs = torch.stack(all_outputs,dim=0).cpu()
    mean = all_outputs.mean(dim=1)
    low_list,high_list = [],[]
    for i in range(len(all_outputs)):
        res = bootstrap((all_outputs[i],), np.mean, confidence_level=0.99,random_state=rng,method="bca",n_resamples=5000)
        low,high = res.confidence_interval.low,res.confidence_interval.high
        low_list.append(low)
        high_list.append(high)

    plt.figure()

    fontsize = 20
    x = np.arange(len(low_list))
    mean_initial_confidence = outputs.mean().cpu()
    plt.plot([0,len(low_list)-1],[mean_initial_confidence,mean_initial_confidence],"--",color="black")
    plt.plot(mean,"-*")
    plt.fill_between(x,low_list,high_list,alpha=0.25)
    plt.xticks(x,kernel_sizes,fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(f"../vis/{args.exp_id}/blur_vs_confidence_{args.model_id}_{args.att_metrics_post_hoc}.png")


if __name__ == "__main__":
    main()

