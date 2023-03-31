import sys
from tkinter import Y
from args import ArgReader
import numpy as np
import sqlite3 
from metrics import krippendorff_alpha_paralel,krippendorff_alpha_bootstrap,get_sub_multi_step_metric_list,get_correlation_metric_list,get_metrics_to_minimize
import matplotlib.pyplot as plt 
from scipy.stats._resampling import _bootstrap_iv,rng_integers,_percentile_of_score,ndtri,ndtr,BootstrapResult,ConfidenceInterval

import warnings
from scipy.stats._warnings_errors import DegenerateDataWarning

from scipy.stats import pearsonr,kendalltau,bootstrap

from does_cumulative_increase_interrater_reliability import fmt_metric_values,preprocc_matrix,fmt_value_str,get_background_func,get_post_hoc_label_dic

from torch.nn.functional import cross_entropy
from torch import from_numpy,arange,softmax

from compute_saliency_metrics import get_db
from is_faithfulness_related_to_class import filter_query_result

def make_comb(dim_nb,dim_ind=None,comb_list=None):

    if comb_list is None:
        comb_list = [[i] for i in range(dim_nb)]
        dim_ind = 0
    
    if dim_ind < dim_nb-1:
        new_comb_list = []
        for i in range(len(comb_list)):
            already_used_items = comb_list[i]
            not_yet_used_items = list(set(list(range(dim_nb))) - set(already_used_items))
            new_comb_list.extend([comb_list[i]+[item] for item in not_yet_used_items])   
        return make_comb(dim_nb,dim_ind+1,new_comb_list)
    else:
        return comb_list

def make_rank_hist(metric_values_matrix):
    sizes = []
    rank_list = np.array(make_comb(4))+1
    sizes = (metric_values_matrix[:,np.newaxis] == rank_list[np.newaxis]).all(axis=2).sum(axis=0)
    rank_list = ["".join(row.astype(str)) for row in rank_list]
    return sizes,rank_list

def add_missing(values,count,explanation_nb):

    if len(values) < explanation_nb:

        missing_values= set(range(1,explanation_nb+1)) - set(values)
        
        values = list(values) + list(missing_values)
        count = list(count) + [0 for _ in range(len(missing_values))]

    return values,count 

def ranking_distribution(metric_values_list,metrics_to_minimize,metric,cumulative_suff,label,expl_names,fig_rank,axs_rank,fig_globrank,axs_globrank):

    lab_dic = get_post_hoc_label_dic()

    metric_values_matrix = fmt_metric_values(metric_values_list)
    if metric not in metrics_to_minimize:
        metric_values_matrix = -metric_values_matrix
    metric_values_matrix = metric_values_matrix.argsort(-1)+1

    explanation_nb = metric_values_matrix.shape[1]

    #Rank distribution
    sizes,names = make_rank_hist(metric_values_matrix)
    if fig_rank is None:
        fig_rank, axs_rank = plt.subplots(2,1,figsize=(20,10))
        fig_globrank,axs_globrank = plt.subplots(2,explanation_nb,figsize=(20,10))

    x = np.arange(len(names))
    row = 1*(cumulative_suff=="")
    axs_rank[row].bar(x,sizes)
    axs_rank[row].set_xticks(x,names,rotation=45,ha="right")
    axs_rank[row].set_title(label)

    x = np.arange(explanation_nb)
    for i in range(explanation_nb): 
        values,count = np.unique(metric_values_matrix[:,i],return_counts=True)
        values,count = add_missing(values,count,explanation_nb)
        values,count = zip(*sorted(zip(values,count),key=lambda x:x[0]))
        count = np.array(count)
        freq = count/count.sum()
        axs_globrank[row,i].bar(x,freq)
        method_labels = [lab_dic[name] if name in lab_dic else name for name in expl_names]
        axs_globrank[row,i].set_xticks(x,method_labels,rotation=45,ha="right")
        axs_globrank[row,i].set_title(label+" - "+"Top "+str(i+1)+" method distribution")
        axs_globrank[row,i].set_ylim(0,1.1)

    return fig_rank, axs_rank,fig_globrank,axs_globrank

#y_delta variance vs saliency rank
def delta_variance_vs_saliency_rank(post_hoc_methods,output_list,saliency_scores_list,output_at_each_step_list,metric,corr_metrics,label,fig,axs):
    
    if fig is None:
        fig,axs = plt.subplots(len(post_hoc_methods)//2,len(post_hoc_methods)//2,figsize=(20,10))

    for i,post_hoc_method in enumerate(post_hoc_methods):
        output = fmt_value_str(output_list[i])
        saliency_scores = fmt_value_str(saliency_scores_list[i])
        output_at_each_step = fmt_value_str(output_at_each_step_list[i])
        
        batch_inds = np.arange(len(output))
        class_inds = output.argmax(axis=-1)
        output = output[batch_inds,class_inds]
        output_at_each_step = output_at_each_step[batch_inds,1:,class_inds]

        delta_output = output[:,np.newaxis] - output_at_each_step

        delta_output_std = delta_output.std(axis=0)
        
        if metric in corr_metrics:
            axs[i//2,i%2].plot(saliency_scores.mean(axis=0),delta_output_std,label=label)
        else:
            axs[i//2,i%2].plot(delta_output_std,label=label)
        axs[i//2,i%2].legend()
        axs[i//2,i%2].set_ylabel("Score variation standard deviation")
        axs[i//2,i%2].set_title(post_hoc_method)
        axs[i//2,i%2].grid()
    return fig,axs
def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--background', type=str)
    argreader.parser.add_argument('--ordinal_metric', action="store_true")
    argreader.parser.add_argument('--compare_models', action="store_true")
    argreader.parser.add_argument('--accepted_models_to_compare',nargs="*",type=str)

    #Reading the comand line arg
    argreader.getRemainingArgs()
    args = argreader.args

    model_id = args.model_id
    exp_id = args.exp_id 

    con,curr = get_db(exp_id)

    corr_metrics = get_correlation_metric_list()
    metrics_to_minimize = get_metrics_to_minimize()
    metrics = get_sub_multi_step_metric_list()
    background_func = get_background_func(args.background)

    for metr_ind,metric in enumerate(metrics):
        print(metric)

        fig, axs = None,None
        fig_rank, axs_rank = None,None
        fig_globrank,axs_globrank = None,None

        for cumulative_suff in ["","-nc"]:
            label="Cumulative" if cumulative_suff == "" else "Non-cumulative"
            print("\t",label)
            background = background_func(metric)

            query = f'SELECT post_hoc_method,metric_value,outputs,saliency_scores,prediction_scores FROM metrics WHERE model_id=="{model_id}" and metric_label=="{metric+cumulative_suff}" and replace_method=="{background}"'
            output = curr.execute(query).fetchall()
            output = filter_query_result(output,[2,3,4])

            if len(output) > 0:
                post_hoc_methods,metric_values_list,output_list,saliency_scores_list,output_at_each_step_list = zip(*output)

                fig_rank, axs_rank,fig_globrank,axs_globrank = ranking_distribution(metric_values_list,metrics_to_minimize,metric,cumulative_suff,label,post_hoc_methods,fig_rank, axs_rank,fig_globrank,axs_globrank)

                fig,axs = delta_variance_vs_saliency_rank(post_hoc_methods,output_list,saliency_scores_list,output_at_each_step_list,metric,corr_metrics,label,fig,axs)
        
        fig.savefig(f"../vis/{exp_id}/yvar_vs_rank_{metric}.png")
        fig_rank.savefig(f"../vis/{exp_id}/rank_{metric}.png")
        fig_globrank.tight_layout()
        fig_globrank.savefig(f"../vis/{exp_id}/globrank_{metric}.png")

if __name__ == "__main__":
    main()