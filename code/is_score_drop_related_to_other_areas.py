import os
import math
from collections import namedtuple

import matplotlib.pyplot as plt

import torch

from args import ArgReader,init_post_hoc_arg,addLossTermArgs
import modelBuilder
import load_data
from saliency_maps_metrics.multi_step_metrics import Deletion
from does_resolution_impact_faithfulness import load_model,get_data_inds_and_explanations


def make_global_salrank_figure(all_dists,all_rank_diff,exp_id,model_id,att_metrics_post_hoc):

    all_dists = torch.cat(all_dists,dim=0)
    all_rank_diff = torch.cat(all_rank_diff,dim=0)

    distance_to_rank_diff_dic = {}

    for i in range(len(all_dists)):

        dist = all_dists[i].item()

        if not dist in distance_to_rank_diff_dic:
            distance_to_rank_diff_dic[dist] = []
        
        distance_to_rank_diff_dic[dist].append(all_rank_diff[i])

    dists = list(sorted(list(distance_to_rank_diff_dic.keys())))
    #dists = dists[:5]
    fig,axs = plt.subplots(len(dists),1,figsize=(5,15))

    for i,dist in enumerate(dists):
        
        axs[i].hist(distance_to_rank_diff_dic[dist])
        axs[i].set_ylabel(dist)

    fig.savefig(f"../vis/{exp_id}/relative_diff_vs_dists_glob_salrank_{model_id}_{att_metrics_post_hoc}.png")

def diff(value):
    return torch.abs(value.reference-value.updated)

def relative_diff(value):
    return torch.abs(value.reference-value.updated)/value.updated

def compute_saliency(retDict,chosen_location_ind,mask_res,predicted_class,weights):
    activations = retDict["feat"]
    weights = weights[predicted_class]
    saliency = (weights.unsqueeze(0).unsqueeze(-1).unsqueeze(-1)*activations).mean(dim=1,keepdim=True)
    print(saliency.shape)
    
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
    all_inds_but_chosen_location = list(set(all_inds) - set([chosen_location_ind]))
    all_inds_but_chosen_location = torch.tensor(all_inds_but_chosen_location).long()
    masks = masks[all_inds_but_chosen_location]
    
    masks_interp = torch.nn.functional.interpolate(masks,data.shape[2],mode="nearest")
    retDict = net(data*masks_interp)
    ref_scores = retDict["output"][:,predicted_class]
    weight = net.secondModel.linLay.weight
    ref_saliency,ref_saliency_rank = compute_saliency(retDict,row,col,predicted_class,weight)

    ratio = data.shape[2]//masks.shape[2]
    masks_interp[:,:,row*ratio:(row+1)*ratio,col*ratio:(col+1)*ratio] = 0
    retDict = net(data*masks_interp)
    updated_scores = retDict["output"][:,predicted_class]
    updated_saliency,updated_saliency_rank = compute_saliency(retDict,row,col,predicted_class,weight)
    
    Value = namedtuple('value', 'reference updated')

    scores = Value(ref_scores,updated_scores)
    saliency = Value(ref_saliency,updated_saliency)
    saliency_rank = Value(ref_saliency_rank,updated_saliency_rank)

    return scores,saliency,saliency_rank

def load_or_compute_scores(exp_id,model_id,net_lambda,data,masks,post_hoc_method):

    result_file_path = f"../results/{exp_id}/scores_with_get_random_masks_{model_id}_{post_hoc_method}.th"

    if not os.path.exists(result_file_path):
        print("Computing masked scores")
        all_scores = []
        with torch.no_grad():
            for i in range(len(data)):
                scores = net_lambda(data[i:i+1]*masks)
                all_scores.append(scores)
        scores = torch.stack(all_scores,dim=0)
        torch.save(scores,result_file_path)
    else:
        scores = torch.load(result_file_path)

    return scores

def get_random_masks(mask_nb,mask_res,device):
    torch.manual_seed(0)
    masks = torch.zeros((mask_nb,1,mask_res,mask_res)).to(device)
    masks = masks.bernoulli_()
    return masks

def get_masks(mask_res,device):
    torch.manual_seed(0)
    #masks = torch.ones((mask_res*mask_res,1,mask_res,mask_res)).to(device)
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

    ######## for debug ################
    debug_ind = 16
    #data = data[:debug_ind]
    #explanations = explanations[:debug_ind]
    #predicted_classes = predicted_classes[:debug_ind]
    ###################################

    fig_global_scatter,ax_global_scatter = plt.subplots(1,1,figsize=(7,5))
    fig_global_scatter_sal,ax_global_scatter_sal = plt.subplots(1,1,figsize=(7,5))
    fig_global_scatter_salrank,ax_global_scatter_salrank = plt.subplots(1,1,figsize=(7,5))

    masks,row_list,col_list = get_masks(explanations.shape[2],data.device)
    row_list,col_list = row_list.to(explanations.device),col_list.to(explanations.device)

    chosen_location_inds = explanations.view(explanations.shape[0],-1).argsort(dim=-1)[:,-1]
    print("got scores_mask !")

    all_rows = torch.arange(explanations.shape[2]).unsqueeze(0).unsqueeze(0).unsqueeze(-1).to(explanations.device)
    all_rows = all_rows.expand(-1,-1,-1,explanations.shape[3])

    all_cols = torch.arange(explanations.shape[3]).unsqueeze(0).unsqueeze(0).unsqueeze(0).to(explanations.device)
    all_cols = all_cols.expand(-1,-1,explanations.shape[2],-1)
    
    all_dists, all_rank_diff = [],[]
    for i in range(len(data)):
        print(i)

        scores,saliency,saliency_rank = compute_score_drop(net,data[i:i+1],chosen_location_inds[i],masks,predicted_classes[i])

        row = chosen_location_inds[i].div(masks.shape[3],rounding_mode='floor')
        col = chosen_location_inds[i] % masks.shape[3]
 
        dists = torch.abs(row_list-row)+torch.abs(col_list-col)
     
        #scores.reference = scores.reference[:,predicted_classes[i]]
        #scores.updated = scores.updated[:,predicted_classes[i]]
        relative_diff_scores = relative_diff(scores)
        ax_global_scatter.scatter(dists.cpu(),relative_diff_scores.cpu(),alpha=0.01,color="blue")

        relative_diff_saliency = relative_diff(saliency)
        ax_global_scatter_sal.scatter(dists.cpu(),relative_diff_saliency.cpu(),alpha=0.01,color="blue")

        diff_saliency_rank = diff(saliency_rank)
        ax_global_scatter_salrank.scatter(dists.cpu(),diff_saliency_rank.cpu(),alpha=0.01,color="blue")

        all_dists.append(dists.cpu())
        all_rank_diff.append(diff_saliency_rank.cpu())

        if i == debug_ind-1:
            break

    ax_global_scatter_sal.set_ylim(0,2.5)
    #ax_global_scatter_salrank.set_ylim(0,3)

    fig_global_scatter.savefig(f"../vis/{args.exp_id}/relative_diff_vs_dists_glob_{args.model_id}_{args.att_metrics_post_hoc}.png")
    fig_global_scatter_sal.savefig(f"../vis/{args.exp_id}/relative_diff_vs_dists_glob_sal_{args.model_id}_{args.att_metrics_post_hoc}.png")
    fig_global_scatter_salrank.savefig(f"../vis/{args.exp_id}/relative_diff_vs_dists_glob_salrank_onefig_{args.model_id}_{args.att_metrics_post_hoc}.png")

    make_global_salrank_figure(all_dists,all_rank_diff,args.exp_id,args.model_id,args.att_metrics_post_hoc)


if __name__ == "__main__":
    main()