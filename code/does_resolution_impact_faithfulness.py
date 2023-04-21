import os
import glob
import math 

import matplotlib.pyplot as plt 
import numpy as np
import torch
from scipy.stats import bootstrap

from args import ArgReader,init_post_hoc_arg,addLossTermArgs
import modelBuilder
import load_data
from init_model import preprocessAndLoadParams
from compute_scores_for_saliency_metrics import get_attr_func,compute_or_load_explanations
from metrics import get_sal_metric_dics,sample_img_inds,getBatch
from saliency_maps_metrics.multi_step_metrics import compute_auc_metric
from compute_saliency_metrics import apply_softmax
import utils 

def get_ylabel(is_auc,metric_name,is_cumulative):

    if metric_name == "Deletion":
        label = "D"
    else:
        label = "I"
    
    if is_auc:
        label += "AUC"
    else:
        label += "C"
    
    if not is_cumulative:
        label += "-NC"

    return label

def compute_var(scores1,metric_name):
    if len(scores1.shape) == 2:
        if metric_name == "Deletion":
            var_scores1 = scores1[:,:-1] - scores1[:,1:] 
        else:
            var_scores1 = scores1[:,1:] - scores1[:,:-1]
        var_scores1 = var_scores1.mean(axis=0)
    else:
        if metric_name == "Deletion":
            var_scores1 = scores1[:-1] - scores1[1:] 
        else:
            var_scores1 = scores1[1:] - scores1[:-1]
    return var_scores1

def plot_scores(is_auc,metric_name,is_cumulative,factor_ind,factor,axs,scores1,scores2):
    row = get_subplot_row(is_auc,metric_name,is_cumulative)
    col = factor_ind
    ax = axs[row,col]
    
    label = get_ylabel(is_auc,metric_name,is_cumulative)

    if col == 0:    
        ax.set_ylabel(label)
    if row == axs.shape[0]-1:
        ax.set_xlabel(factor)
    
    if not is_auc:
        ax.set_xlim(0,1)

    if is_auc:
        lower_y_lim = 0
        upper_y_lim = 1.2
    else:
        lower_y_lim = -0.15
        upper_y_lim = 0.15

    ax.set_ylim(lower_y_lim,upper_y_lim)
    
    if len(scores2.shape) == 2:
        scores2 = scores2.mean(axis=0)

    if is_auc:
        if len(scores1.shape)==2: 
            scores1 = scores1.mean(axis=0)

        ax.plot(scores1[1:])
        ax.plot([0,len(scores1)-1],[scores1[0],scores1[0]],"--",color="black")
    else:  
        var_scores1 = compute_var(scores1,metric_name)
        ax.scatter(scores2,var_scores1)

def get_subplot_row(is_auc,metric_name,is_cumulative):
    row = 4*is_auc
    row += 2*(metric_name == "Insertion")
    row += 1*is_cumulative
    return row

def get_data_inds_and_explanations(net,net_lambda,testDataset,args):
    attrFunc,kwargs = get_attr_func(net,testDataset,args)

    inds = sample_img_inds(args.img_nb_per_class,testDataset=testDataset)

    data,_ = getBatch(testDataset,inds,args)

    torch.set_grad_enabled(False)
    outputs = net_lambda(data)
    predClassInds = outputs.argmax(dim=-1)

    explanations = compute_or_load_explanations(inds,args,data,predClassInds,attrFunc,kwargs)

    return data,explanations,predClassInds,outputs,inds

def load_model(args):
    bestPath = glob.glob(f"../models/{args.exp_id}/model{args.model_id}_best_epoch*")[0]
    net = modelBuilder.netBuilder(args)
    net = preprocessAndLoadParams(bestPath,args.cuda,net,verbose=False)
    net.eval()
    def net_lambda(x):
        return net(x)["output"]
    return net,net_lambda

def save_data_masked(data_masked,result_file_path):
    step_size = 1
    inds = np.arange(0,len(data_masked),step_size)
    data_masked = data_masked[inds]
    print(data_masked.shape)
    utils.save_image(data_masked,result_file_path.replace("results","vis").replace(".npy",".png"))

def get_row_and_col(sub_metric):
    
    if "-NC" in sub_metric:
        row_ind = 1 
        sub_metric = sub_metric.replace("-NC","")
    else:
        row_ind = 0

    metric_order = {"DAUC":0,"IAUC":1,"DC":2,"IC":3,"IIC":4,"AD":5,"ADD":6}

    col_ind = metric_order[sub_metric.upper()]
    return row_ind,col_ind

def get_metric_and_is_cumulative_lists(const_dic):

    all_metrics = list(const_dic.keys())
    
    metric_list = all_metrics
    suff_list = [True for _ in range(len(all_metrics))]

    metric_list += ["Deletion","Insertion"]
    suff_list += [False,False]

    return metric_list,suff_list
            
def compute_or_load_scores(metric,metric_name,metric_args,args,formated_attention_metric,data_replace_method,post_hoc_suff,scale_suff,is_multi_step_dic,kwargs):

        result_file_path = f"../results/{args.exp_id}/resolution_vs_faith_{formated_attention_metric}-{data_replace_method}_{args.model_id}{post_hoc_suff}_{scale_suff}.npy"
           
        if not os.path.exists(result_file_path):
            if is_multi_step_dic[metric_name]:  
                scores1,scores2,data_masked = metric.compute_scores(*metric_args,**kwargs)
            else:
                scores1,scores2,data_masked = metric.compute_scores(*metric_args,**kwargs)
            np.save(result_file_path,{"scores1":scores1,"scores2":scores2})
            save_data_masked(data_masked,result_file_path)
        else:
            pickle_dict = np.load(result_file_path,allow_pickle=True).item()
            scores1,scores2 = pickle_dict["scores1"],pickle_dict["scores2"]

        return scores1,scores2
            
def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)
    argreader = load_data.addArgs(argreader)
    argreader = modelBuilder.addArgs(argreader)
    argreader = init_post_hoc_arg(argreader)
    argreader = addLossTermArgs(argreader)
    argreader.parser.add_argument('--img_to_show_scores_nb', type=int,default=2)

    #Reading the comand line arg
    argreader.getRemainingArgs()

    #Getting the args from command line and config file
    args = argreader.args

    args.cuda = args.cuda and torch.cuda.is_available()

    is_multi_step_dic,const_dic = get_sal_metric_dics()
    
    rng = np.random.default_rng(0)

    metric_list,is_cumulative_list = get_metric_and_is_cumulative_lists(const_dic)

    _,testDataset = load_data.buildTestLoader(args, "test")

    net,net_lambda = load_model(args)

    data,explanations,predClassInds,_,_ = get_data_inds_and_explanations(net,net_lambda,testDataset,args)

    if "ablationcam" in args.att_metrics_post_hoc:
        #Reloading model to remove hook put by ablationcam module
        net,net_lambda = load_model(args)

    ratio = net.firstModel.featMod.downsample_ratio
    powers = np.arange(math.log(ratio,2)+1)
    scale_factors = np.power(2,powers).astype(int)

    post_hoc_suff = "" if args.att_metrics_post_hoc is None else "-"+args.att_metrics_post_hoc

    global_dict = {}

    image_inds_to_show_score = torch.arange(0,len(data),len(data)//args.img_to_show_scores_nb)
    fig_dict,axs_dict = {},{}
    for ind in image_inds_to_show_score:
        fig,axs = plt.subplots(8,len(scale_factors),figsize=(15,15))    
        fig_dict[ind.item()] = fig
        axs_dict[ind.item()] = axs
    
    fig,axs = plt.subplots(8,len(scale_factors),figsize=(15,15))      
    fig_dict["mean"] = fig
    axs_dict["mean"] = axs   

    for metric_name,is_cumulative in zip(metric_list,is_cumulative_list):
        print(metric_name,is_cumulative)

        formated_attention_metric = metric_name
        #Constructing metric

        metric_constr_arg_dict = {}

        if is_multi_step_dic[metric_name]:
            metric_constr_arg_dict.update({"max_step_nb":explanations.shape[2]*explanations.shape[3],"batch_size":args.val_batch_size})

        if is_multi_step_dic[metric_name]:
            metric_constr_arg_dict.update({"cumulative":is_cumulative})
            if not is_cumulative:
                formated_attention_metric += "nc"

        metric = const_dic[metric_name](**metric_constr_arg_dict)
        
        for factor_ind,factor in enumerate(scale_factors):
            print("\t",factor)

            explanations_resc = torch.nn.functional.interpolate(explanations,scale_factor=factor,mode="bicubic")            
        
            metric_args = [net_lambda,data,explanations_resc,predClassInds]
            kwargs = {"save_all_class_scores":True,"return_data":True}

            scores1,scores2 = compute_or_load_scores(metric,metric_name,metric_args,args,formated_attention_metric,metric.data_replace_method,post_hoc_suff,factor,is_multi_step_dic,kwargs)

            scores1 = apply_softmax(scores1,args.temperature)

            if is_multi_step_dic[metric_name]:
                auc_metric = compute_auc_metric(scores1)        
                calibration_metric = metric.compute_calibration_metric(scores1, scores2)
                result_dic = metric.make_result_dic(auc_metric,calibration_metric)

                for ind in image_inds_to_show_score:
                    fig,axs = fig_dict[ind.item()],axs_dict[ind.item()]          
                    plot_scores(False,metric_name,is_cumulative,factor_ind,factor,axs,scores1[ind],scores2[ind])
                    plot_scores(True,metric_name,is_cumulative,factor_ind,factor,axs,scores1[ind],scores2[ind])

                axs = axs_dict["mean"]
                plot_scores(False,metric_name,is_cumulative,factor_ind,factor,axs,scores1,scores2)
                plot_scores(True,metric_name,is_cumulative,factor_ind,factor,axs,scores1,scores2)
                
            else:
                scores2 = apply_softmax(scores2,args.temperature)
                result_dic = metric.compute_metric(scores1,scores2)

            for sub_metric in result_dic:

                sub_metric_and_cum_suff = sub_metric+"-NC" if not is_cumulative else sub_metric

                if sub_metric_and_cum_suff not in global_dict:
                    global_dict[sub_metric_and_cum_suff] = {"mean":{},"conf_interv_low":{},"conf_interv_high":{}}
                
                mean = result_dic[sub_metric].mean()
                res = bootstrap((result_dic[sub_metric],), np.mean, confidence_level=0.99,random_state=rng,method="bca",n_resamples=5000)
                low,high = res.confidence_interval.low,res.confidence_interval.high

                global_dict[sub_metric_and_cum_suff]["mean"][factor] = mean
                global_dict[sub_metric_and_cum_suff]["conf_interv_low"][factor] = low
                global_dict[sub_metric_and_cum_suff]["conf_interv_high"][factor] = high

    plot_nb = len(global_dict.keys())
    nb_rows = int(math.sqrt(plot_nb))
    nb_cols = plot_nb//nb_rows + 1*(plot_nb%nb_rows>0)

    fig, axs = plt.subplots(nb_rows,nb_cols,figsize=(15,10))
    fontsize = 17
    for i,sub_metric in enumerate(list(global_dict)):
        ax = axs[i//nb_cols,i%nb_cols]

        sub_metric_stats = global_dict[sub_metric]

        factors,metric_values = zip(*sub_metric_stats["mean"].items())
        ax.plot(metric_values,"-*")

        conf_interv_low = list(sub_metric_stats["conf_interv_low"].values())
        conf_inter_high = list(sub_metric_stats["conf_interv_high"].values())

        ax.fill_between(np.arange(len(factors)),conf_interv_low,conf_inter_high,alpha=0.5)

        ylow = min(conf_interv_low)-0.1*abs(min(conf_interv_low))   
        yhigh = max(conf_inter_high)+0.1*abs(max(conf_inter_high))
        ax.set_ylim(ylow,yhigh)
        
        ax.set_xlabel("Factor",fontsize=fontsize)
        ax.set_xticks(np.arange(len(factors)))
        ax.set_xticklabels(factors,fontsize=fontsize)
        ax.tick_params(axis='both', which='minor', labelsize=fontsize)
        ax.tick_params(axis='both', which='major', labelsize=fontsize)
        ax.set_title(sub_metric.upper(),fontsize=fontsize)

    plt.tight_layout()
    plt.savefig(f"../vis/{args.exp_id}/resolution_vs_faithfulness_{args.model_id}_{args.att_metrics_post_hoc}.png")
    plt.close()

    for i,ind in enumerate(image_inds_to_show_score):
        fig,axs = fig_dict[ind.item()],axs_dict[ind.item()]  
        fig.tight_layout() 
        fig.savefig(f"../vis/{args.exp_id}/resolution_vs_faithfulness_{args.model_id}_{args.att_metrics_post_hoc}_img{i}.png")

    fig = fig_dict["mean"]
    fig.tight_layout() 
    fig.savefig(f"../vis/{args.exp_id}/resolution_vs_faithfulness_{args.model_id}_{args.att_metrics_post_hoc}_imgAll.png")

if __name__ == "__main__":
    main()