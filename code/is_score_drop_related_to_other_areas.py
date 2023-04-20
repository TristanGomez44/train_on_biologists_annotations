import os,sys
import math
from collections import namedtuple

import matplotlib.pyplot as plt

import torch
import numpy as np 

from args import ArgReader,init_post_hoc_arg,addLossTermArgs
import modelBuilder
import load_data
from saliency_maps_metrics.multi_step_metrics import Deletion
from does_resolution_impact_faithfulness import load_model,get_data_inds_and_explanations


def load_or_compute_scores(net,data,chosen_location_inds,masks,predicted_classes,result_path):
    
    
    if not os.path.exists(result_path):
        all_scores,all_saliency,all_saliency_rank = [],[],[]
        for i in range(len(data)):
            if i %10== 0:
                print(i,"/",len(data))

            scores,saliency,saliency_rank = compute_score_drop(net,data[i:i+1],chosen_location_inds[i],masks,predicted_classes[i])
            all_scores.append(scores)
            all_saliency.append(saliency)
            all_saliency_rank.append(saliency_rank)

        dic = {"scores":all_scores,"saliency":all_saliency,"saliency_rank":all_saliency_rank}

        np.save(result_path,dic)

    else:
        dic = np.load(result_path,allow_pickle=True).item()
        all_scores,all_saliency,all_saliency_rank = dic["scores"],dic["saliency"],dic["saliency_rank"]

    return all_scores,all_saliency,all_saliency_rank    

def make_dist_to_diff_dic(all_dists,all_diff):
    distance_to_diff_dic = {}
    for i in range(len(all_dists)):
        dist = all_dists[i].item()
        if dist not in distance_to_diff_dic:
            distance_to_diff_dic[dist] = []
        distance_to_diff_dic[dist].append(all_diff[i].item())
    return distance_to_diff_dic

def make_global_salrank_figure(all_dists,all_diff,exp_id,model_id,att_metrics_post_hoc):

    distance_to_diff_dic = make_dist_to_diff_dic(all_dists,all_diff)

    dists = list(sorted(list(distance_to_diff_dic.keys())))
    fig,axs = plt.subplots(len(dists),1,figsize=(5,15))

    for i,dist in enumerate(dists):
        axs[i].hist(distance_to_diff_dic[dist])
        axs[i].set_ylabel(dist)

    fig.savefig(f"../vis/{exp_id}/relative_diff_vs_dists_glob_salrank_{model_id}_{att_metrics_post_hoc}.png")

def median_plot(distance_to_diff_dic,file_path):

    dists = list(sorted(list(distance_to_diff_dic.keys())))
    fig_median,ax_median = plt.subplots(1,1)
    median_list = []
    q1_list,q2_list = [],[]
    for i,dist in enumerate(dists):
        median_list.append(np.median(distance_to_diff_dic[dist]))
        q1_list.append(np.quantile(distance_to_diff_dic[dist],0.25))
        q2_list.append(np.quantile(distance_to_diff_dic[dist],0.75))
    
    ax_median.plot(dists,median_list)
    ax_median.fill_between(dists,q1_list,q2_list,alpha=0.25)
    fig_median.savefig(file_path)

def init_bins(distance_to_diff_dic,nb_bins):
    distance_to_bins_dic = {}
    for dist in distance_to_diff_dic:
        distance_to_bins_dic[dist] = {i:0 for i in range(nb_bins)}
    return distance_to_bins_dic

def make_sub_bin_dict(bin_values,bin_counts,bins):
    sub_bin_dict = {}
    sub_bin_dict = {value:count for value,count in zip(bin_values,bin_counts)}
    for value in bins:
        if not value in sub_bin_dict:
            sub_bin_dict[value] = 0   
    return sub_bin_dict

def add_missing(bin_values,bin_counts,bins):
    bin_counts_with_missing = []
    for bin_value in bins:
        if bin_value in bin_values:
            ind = np.argwhere(bin_values==bin_value)[0][0] 
            bin_counts_with_missing.append(bin_counts[ind])
        else:
            bin_counts_with_missing.append(0)

    return np.array(bin_counts_with_missing)

def make_bins(distance_to_diff_dic,min_value,max_value,nb_bins,limits):

    distance_to_bins_dic = {}
    step_size = (max_value-min_value)/nb_bins
    bins = np.arange(min_value,max_value+step_size,step_size)

    for dist in distance_to_diff_dic.keys():
        diffs = np.array(distance_to_diff_dic[dist])
        bef = diffs.shape
        diffs = diffs[(limits[0] <= diffs)*(diffs<=limits[1])]

        bin_indices = np.digitize(diffs,bins) - 1
        bin_values = bins[bin_indices]
        bin_values,bin_counts = np.unique(bin_values,return_counts=True)
        bin_counts = add_missing(bin_values,bin_counts,bins)
        distance_to_bins_dic[dist] = (bin_counts-bin_counts.min())/(bin_counts.max()-bin_counts.min())

    return distance_to_bins_dic,step_size,bins

def density_plot(distance_to_diff_dic,all_diff,file_path,nb_bins=50,ylim=None):
    
    if ylim is None:
        min_value,max_value = all_diff.min(),all_diff.max()
    else:
        min_value,max_value = ylim[0],ylim[1]

    distance_to_bins_dic,step_size,bins = make_bins(distance_to_diff_dic,min_value,max_value,nb_bins,ylim)
    dists = list(sorted(list(distance_to_diff_dic.keys())))
    fig,ax = plt.subplots(1,1,figsize=(15,15))
    cmap = plt.get_cmap("plasma")

    fontsize=25

    for i,dist in enumerate(dists):
        x = np.array([dist]).repeat(nb_bins+1,0)
        h = np.array([step_size]).repeat(nb_bins+1,0)
        colors = cmap(distance_to_bins_dic[dist])[:,:3]

        ax.bar(x,h,1,bins,color=colors)

        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.tick_params(axis='both', which='minor', labelsize=fontsize)

        if not ylim is None:
            ax.set_ylim(ylim[0],ylim[1])

    fig.savefig(file_path)

def diff(value):
    return torch.abs(value["ref"]-value["updated"])

def relative_diff(value):
    return torch.abs(value["ref"]-value["updated"])/value["ref"]

def compute_saliency(retDict,chosen_location_ind,mask_res,predicted_class,weights):
    activations = retDict["feat"]
    weights = torch.relu(weights[predicted_class])
    saliency = (weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)*activations).mean(dim=1,keepdim=True)
    saliency = (saliency-saliency.min())/(saliency.max()-saliency.min())
    row = chosen_location_ind.div(mask_res,rounding_mode='floor')
    col = chosen_location_ind % mask_res
    saliency_ranks = (-saliency.view(saliency.shape[0],-1)).argsort(dim=-1)+1
    saliency_rank = saliency_ranks[:,chosen_location_ind]
    saliency = saliency[:,0,row,col]
    return saliency,saliency_rank

def compute_score_drop(net,data,chosen_location_ind,masks,predicted_class):

    row = chosen_location_ind.div(masks.shape[3],rounding_mode='floor')
    col = chosen_location_ind % masks.shape[3]

    all_inds = list(range(len(masks)))
    all_inds_but_chosen_location = list(set(all_inds) - set([chosen_location_ind.item()]))
    all_inds_but_chosen_location = torch.tensor(all_inds_but_chosen_location).long()
    
    masks = masks[all_inds_but_chosen_location]

    masks_interp = torch.nn.functional.interpolate(masks,data.shape[2],mode="nearest")
    retDict = net(data*masks_interp)
    ref_scores = retDict["output"][:,predicted_class]
    weight = net.secondModel.linLay.weight
    ref_saliency,ref_saliency_rank = compute_saliency(retDict,chosen_location_ind,masks.shape[3],predicted_class,weight)

    ratio = data.shape[2]//masks.shape[2]
    masks_interp[:,:,row*ratio:(row+1)*ratio,col*ratio:(col+1)*ratio] = 0
    retDict = net(data*masks_interp)
    updated_scores = retDict["output"][:,predicted_class]
    updated_saliency,updated_saliency_rank = compute_saliency(retDict,chosen_location_ind,masks.shape[3],predicted_class,weight)
    
    scores = {"ref":ref_scores.cpu(),"updated":updated_scores.cpu()}
    saliency = {"ref":ref_saliency.cpu(),"updated":updated_saliency.cpu()}
    saliency_rank = {"ref":ref_saliency_rank.cpu(),"updated":updated_saliency_rank.cpu()}

    return scores,saliency,saliency_rank

def get_random_masks(mask_nb,mask_res,device):
    torch.manual_seed(0)
    masks = torch.zeros((mask_nb,1,mask_res,mask_res)).to(device)
    masks = masks.bernoulli_()
    return masks

def get_masks(mask_res,device):
    torch.manual_seed(0)
    mask_template = torch.ones((1,1,mask_res,mask_res)).to(device)
    
    mask_list = []
    row_list,col_list = [],[]

    nb_cols = mask_res
    for i in range(mask_res*mask_res):
        mask = mask_template.clone()
        row,col = i//nb_cols,i%nb_cols
        mask[0,0,row,col] = 0
        mask_list.append(mask)
        row_list.append(row),col_list.append(col)
    mask_list = torch.cat(mask_list,dim=0)

    row_list,col_list = torch.tensor(row_list),torch.tensor(col_list)

    return mask_list,row_list,col_list

def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)
    argreader = load_data.addArgs(argreader)
    argreader = modelBuilder.addArgs(argreader)
    argreader = init_post_hoc_arg(argreader)
    argreader = addLossTermArgs(argreader)
    argreader.parser.add_argument('--mask_nb', type=int,default=100)
    #Reading the comand line arg
    argreader.getRemainingArgs()
   
    #Getting the args from command line and config file
    args = argreader.args

    net,net_lambda = load_model(args)
    _,testDataset = load_data.buildTestLoader(args, "test")
    data,explanations,predicted_classes,scores = get_data_inds_and_explanations(net,net_lambda,testDataset,args)

    debug_ind = 101

    fig_global_scatter,ax_global_scatter = plt.subplots(1,1,figsize=(7,5))
    fig_global_scatter_sal,ax_global_scatter_sal = plt.subplots(1,1,figsize=(7,5))
    fig_global_scatter_salrank,ax_global_scatter_salrank = plt.subplots(1,1,figsize=(7,5))

    masks,row_list,col_list = get_masks(explanations.shape[2],data.device)
    row_list,col_list = row_list.to(explanations.device),col_list.to(explanations.device)

    chosen_location_inds = explanations.view(explanations.shape[0],-1).argsort(dim=-1)[:,-1]

    all_rows = torch.arange(explanations.shape[2]).unsqueeze(0).unsqueeze(0).unsqueeze(-1).to(explanations.device)
    all_rows = all_rows.expand(-1,-1,-1,explanations.shape[3])

    all_cols = torch.arange(explanations.shape[3]).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(explanations.device)
    all_cols = all_cols.expand(-1,-1,explanations.shape[2],-1)
    
    all_dists, all_diff = [],{"scores":[],"saliency":[],"saliency_rank":[]}

    result_file_path = f"../results/{args.exp_id}/relative_diff_vs_dists_{args.model_id}_{args.att_metrics_post_hoc}.npy"
    all_scores,all_saliency,all_saliency_rank = load_or_compute_scores(net,data,chosen_location_inds,masks,predicted_classes,result_file_path)

    for i in range(len(data)):

        scores,saliency,saliency_rank = all_scores[i],all_saliency[i],all_saliency_rank[i]

        row = chosen_location_inds[i].div(masks.shape[3],rounding_mode='floor')
        col = chosen_location_inds[i] % masks.shape[3]
 
        dists = (torch.abs(row_list-row)+torch.abs(col_list-col)).cpu()
        dists = dists[dists!=0]

        diff_scores = relative_diff(scores).cpu()
        ax_global_scatter.scatter(dists,diff_scores,alpha=0.0025,color="blue")

        diff_saliency = relative_diff(saliency).cpu()
        ax_global_scatter_sal.scatter(dists,diff_saliency,alpha=0.0025,color="blue")

        diff_saliency_rank = diff(saliency_rank).cpu()
        ax_global_scatter_salrank.scatter(dists,diff_saliency_rank,alpha=0.0025,color="blue")

        all_dists.append(dists.cpu())
        all_diff["scores"].append(diff_scores)
        all_diff["saliency"].append(diff_saliency)
        all_diff["saliency_rank"].append(diff_saliency_rank)
   
        if i == debug_ind-1:
            break

    all_dists = torch.cat(all_dists,dim=0)
    for key in all_diff:
        all_diff[key] = torch.cat(all_diff[key],dim=0)
    ax_global_scatter_sal.set_ylim(0,2.5)

    fig_global_scatter.savefig(f"../vis/{args.exp_id}/relative_diff_vs_dists_glob_{args.model_id}_{args.att_metrics_post_hoc}.png")
    fig_global_scatter_sal.savefig(f"../vis/{args.exp_id}/relative_diff_vs_dists_glob_sal_{args.model_id}_{args.att_metrics_post_hoc}.png")
    fig_global_scatter_salrank.savefig(f"../vis/{args.exp_id}/relative_diff_vs_dists_glob_salrank_onefig_{args.model_id}_{args.att_metrics_post_hoc}.png")

    make_global_salrank_figure(all_dists,all_diff["saliency_rank"],args.exp_id,args.model_id,args.att_metrics_post_hoc)

    ylim_dic = {"scores":(0,0.175),"saliency":(0,1.2),"saliency_rank":(0,80)}
    nb_bins_dic = {"scores":50,"saliency":50,"saliency_rank":10}

    for key in all_diff.keys():
        distance_to_diff_dic = make_dist_to_diff_dic(all_dists,all_diff[key]) 
        result_file_path = f"../vis/{args.exp_id}/relative_diff_vs_dists_{key}_median_{args.model_id}_{args.att_metrics_post_hoc}.png"
        median_plot(distance_to_diff_dic,result_file_path)
        density_plot(distance_to_diff_dic,all_diff[key],result_file_path.replace("median","heatmap"),ylim=ylim_dic[key],nb_bins=nb_bins_dic[key])

if __name__ == "__main__":
    main()