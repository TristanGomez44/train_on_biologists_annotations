from tkinter import Y
from args import ArgReader
import numpy as np
import sqlite3 
from metrics import krippendorff_alpha_paralel,krippendorff_alpha_bootstrap
import matplotlib.pyplot as plt 
from scipy.stats._resampling import _bootstrap_iv,rng_integers,_percentile_of_score,ndtri,ndtr,BootstrapResult,ConfidenceInterval

import warnings
from scipy.stats._warnings_errors import DegenerateDataWarning

from scipy.stats import pearsonr,kendalltau,bootstrap

from does_cumulative_increase_interrater_reliability import fmt_metric_values,preprocc_matrix,fmt_value_str

from torch.nn.functional import cross_entropy
from torch import from_numpy,arange,softmax
def filter_query_result(result):

    result = list(filter(lambda x:x[0] != "",result))

    for i in [2,3,4,5]:
        result = list(filter(lambda x:x[i] != "None",result))
    return result

def target_vs_metric(explanation_names,metric_values_matrix,target_list,exp_id,metric):
    fig, axs = plt.subplots(1,len(explanation_names),figsize=(20,10))
    if len(explanation_names) == 1:
        axs = np.array([axs])

    class_nb = len(set(target_list))
    rng = np.random.default_rng(0)

    for i in range(len(explanation_names)):

        name = explanation_names[i]
        metric_val = metric_values_matrix[:,i]
        
        metric_val_per_class = []
        for j in range(class_nb):

            mean = metric_val[target_list==j].mean()
            res = bootstrap((metric_val[target_list==j],), np.mean, confidence_level=0.99,random_state=rng,method="bca")
            low,high = res.confidence_interval.low,res.confidence_interval.high
            low = abs(low-mean)
            high = abs(high-mean)

            metric_val_per_class.append((j,mean,low,high))

        #metric_val_per_class = [(j,metric_val[target_list==j].mean()) for j in range(class_nb)]
        metric_val_per_class = sorted(metric_val_per_class,key=lambda x:x[1])
        metric_val_per_class = np.array(metric_val_per_class)
        x = np.arange(class_nb)
        y = metric_val_per_class[:,1]
        yerr = metric_val_per_class[:,2:].transpose(1,0)

        axs[i].bar(x,y)
        axs[i].errorbar(x,y,yerr,color="black")
        axs[i].set_xticks(x,metric_val_per_class[:,0])
        axs[i].set_title(name)
        axs[i].set_ylim(metric_values_matrix.min()-0.1,metric_values_matrix.max()+0.1)

    plt.savefig(f"../vis/{exp_id}/relationship_class_vs_{metric}.png")
    plt.close()

def correctness_vs_metric(explanation_names,metric_values_matrix,correct_list,exp_id,metric):
    fig, axs = plt.subplots(1,len(explanation_names),figsize=(20,10))

    if len(explanation_names) == 1:
        axs = np.array([axs])
    
    min_val,max_val= metric_values_matrix.min()-0.1,metric_values_matrix.max()+0.1

    for i in range(len(explanation_names)):

        name = explanation_names[i]
        metric_val = metric_values_matrix[:,i]
        
        for correct in [True,False]:
            axs[i].hist(metric_val[correct_list==correct],label=correct,alpha=0.5,density=True,bins=10,range=(min_val,max_val))
    
        axs[i].legend()
        axs[i].set_title(name)
        axs[i].set_xlim(min_val,max_val)

    plt.savefig(f"../vis/{exp_id}/relationship_correctness_vs_{metric}.png")
    plt.close()

def loss_vs_metric(explanation_names,metric_values_matrix,loss_list,target_list,exp_id,metric,cmap):

    fig, axs = plt.subplots(1,len(explanation_names),figsize=(20,10))
    if len(explanation_names) == 1:
        axs = np.array([axs])
    
    for i in range(len(explanation_names)):
        name = explanation_names[i]
        metric_val = metric_values_matrix[:,i]    
        colors = cmap(target_list/target_list.max())
        axs[i].scatter(loss_list,metric_val,alpha=0.5,color=colors)
        axs[i].set_title(name)
        axs[i].set_ylabel(metric)
        axs[i].set_xlabel("Cross entropy loss")
        axs[i].set_ylim(metric_values_matrix.min()-0.1,metric_values_matrix.max()+0.1)
        axs[i].set_xlim(loss_list.min()-0.1,loss_list.max()+0.1)

    plt.savefig(f"../vis/{exp_id}/relationship_loss_vs_{metric}.png")
    plt.close()

def score_vs_metric(explanation_names,metric_values_matrix,output_list,target_list,exp_id,metric,cmap):

    fig, axs = plt.subplots(1,len(explanation_names),figsize=(20,10))
    if len(explanation_names) == 1:
        axs = np.array([axs])
    
    for i in range(len(explanation_names)):
        name = explanation_names[i]
        metric_val = metric_values_matrix[:,i]    
        colors= cmap(target_list/target_list.max())
        axs[i].scatter(output_list,metric_val,alpha=0.5,color=colors)
        axs[i].set_title(name)
        axs[i].set_ylabel(metric)
        axs[i].set_xlabel("Output score")
        axs[i].set_ylim(metric_values_matrix.min()-0.1,metric_values_matrix.max()+0.1)
        axs[i].set_xlim(output_list.min()-0.1,output_list.max()+0.1)

    plt.savefig(f"../vis/{exp_id}/relationship_output_vs_{metric}.png")
    plt.close()

def sparsity_vs_metric(explanation_names,metric_values_matrix,sparsity,target_list,exp_id,metric,cmap):

    fig, axs = plt.subplots(1,len(explanation_names),figsize=(20,10))
    if len(explanation_names) == 1:
        axs = np.array([axs])
    
    for i in range(len(explanation_names)):
        name = explanation_names[i]
        metric_val = metric_values_matrix[:,i]    
        colors= cmap(target_list/target_list.max())
        axs[i].scatter(sparsity,metric_val,alpha=0.5,color=colors)
        axs[i].set_title(name)
        axs[i].set_ylabel(metric)
        axs[i].set_xlabel("Sparsity")
        axs[i].set_ylim(metric_values_matrix.min()-0.1,metric_values_matrix.max()+0.1)
        axs[i].set_xlim(sparsity.min()-0.1,sparsity.max()+0.1)

    plt.savefig(f"../vis/{exp_id}/relationship_sparsity_vs_{metric}.png")
    plt.close()      

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

    #Getting the args from command line and config file
    args = argreader.args

    exp_id = "CROHN25"

    model_id = args.model_id

    if args.compare_models:
        if args.accepted_models_to_compare is None:
            accepted_models = ["noneRed2","noneRed2_transf","noneRed_focal2","noneRed_focal2_transf"]
        else:
            accepted_models = args.accepted_models_to_compare
    else:
        accepted_models = None

    single_step_metrics = ["IIC","AD","ADD"]
    multi_step_metrics = ["DAUC","DC","IAUC","IC"]
    metrics_to_minimize = ["DAUC","AD"]
    metrics = multi_step_metrics+single_step_metrics

    if args.background is None:
        background_func = lambda x:"blur" if x in ["IAUC","IC","INSERTION_VAL_RATE"] else "black"
    else:
        background_func = lambda x:args.background

    db_path = f"../results/{exp_id}/saliency_metrics.db"
   
    con = sqlite3.connect(db_path) # change to 'sqlite:///your_filename.db'
    cur = con.cursor()

    csv_krippen = "cumulative,"+ ",".join(metrics) + "\n"

    explanation_names_list = []
    inds_lists = []

    if args.compare_models:
        filename_suff = f"models"
    else:
        filename_suff = f"{model_id}_b{args.background}"

    cmap = plt.get_cmap("Set1")

    for cumulative_suff in ["","-nc"]:

        csv_krippen += "False" if cumulative_suff == "-nc" else "True"

        for metr_ind,metric in enumerate(metrics):
            
            print(metric,cumulative_suff)

            if metric not in single_step_metrics or cumulative_suff=="": 
                background = background_func(metric)
                metric += cumulative_suff

                query = f'SELECT post_hoc_method,metric_value,outputs,target,inds,saliency_scores FROM metrics WHERE model_id=="{model_id}" and metric_label=="{metric}" and replace_method=="{background}"'
                
                output = cur.execute(query).fetchall()
                
                output = filter_query_result(output)
                
                if len(output) > 0:
                    explanation_names,metric_values_list,output_list,target_list,inds,saliency_scores_list = zip(*output)

                    metric_values_matrix = fmt_metric_values(metric_values_list)

                    output_list = fmt_value_str(output_list[0])
                    target_list = fmt_value_str(target_list[0])
                    inds = fmt_value_str(inds[0])
                    saliency_scores_list = fmt_value_str(saliency_scores_list[0])

                    pred_list = output_list.argmax(axis=-1)
                    correct_list = (pred_list==target_list)
                    correctness_vs_metric(explanation_names,metric_values_matrix,correct_list,exp_id,metric)

                    target_vs_metric(explanation_names,metric_values_matrix,target_list,exp_id,metric)

                    output_list,target_list = from_numpy(output_list), from_numpy(target_list).long()
                    loss_list = cross_entropy(output_list,target_list,reduction="none").numpy()
                    loss_vs_metric(explanation_names,metric_values_matrix,loss_list,target_list,exp_id,metric,cmap)

                    output_list = softmax(output_list,dim=-1)[arange(len(target_list)),target_list]
                    score_vs_metric(explanation_names,metric_values_matrix,output_list,target_list,exp_id,metric,cmap)

                    saliency_scores_list = saliency_scores_list.reshape(saliency_scores_list.shape[0],-1)
                    sparsity = saliency_scores_list.max(axis=-1)/saliency_scores_list.mean(axis=-1)
                    sparsity_vs_metric(explanation_names,metric_values_matrix,sparsity,target_list,exp_id,metric,cmap)
         
                    explanation_names_list.append(explanation_names)
                    inds_lists.append(tuple(inds))

    for parameter_list,parameter_name in zip([explanation_names_list,inds_lists],["explanations","inds"]):
        parameter_set = set(parameter_list)
        if len(parameter_set) != 1:
            print(f"Different sets of {parameter_name} methods were used:{parameter_set}")

if __name__ == "__main__":
    main()