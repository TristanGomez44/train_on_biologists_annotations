
from args import ArgReader
import os,sys
import glob
import numpy as np
import torch 
import matplotlib.pyplot as plt 

from metrics import get_sal_metric_dics,add_validity_rate_multi_step,add_validity_rate_single_step
from saliency_maps_metrics.multi_step_metrics import compute_auc_metric
from modelBuilder import addArgs as addArgsModelBuilder
from compute_saliency_metrics import get_score_file_paths
import sqlite3 ,csv as csvLib,os,sys
from scipy.stats import pearsonr,kendalltau,bootstrap
from does_cumulative_increase_interrater_reliability import get_background_func,addArgs as addArgsDoesCum
from compute_saliency_metrics import get_db,get_info
from metrics import get_sal_metric_dics
from is_faithfulness_related_to_class import filter_query_result,fmt_metric_values,fmt_value_str
from sklearn.neural_network import MLPRegressor
from sklearn import svm

def train_eval(maps,target,n,model,rng):
    x_train,x_test = maps[:n//2],maps[n//2:]
        
    y_train,y_test = target[:n//2],target[n//2:]

    np.random.seed(0)
    model.fit(x_train,y_train)
    y_pred_test = model.predict(x_test)

    err = np.abs(y_pred_test-y_test)/y_test
    mean = err.mean()
    res = bootstrap((err,), np.mean, confidence_level=0.99,random_state=rng,method="bca")
    low,high = res.confidence_interval.low,res.confidence_interval.high
    low = abs(low-mean)
    high = abs(high-mean)

    return mean,low,high

def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--model_ids', type=str,nargs="*", help='Authorized model IDs. Do not set this arg to authorize all model.')

    argreader = addArgsModelBuilder(argreader)
    argreader = addArgsDoesCum(argreader)

    #Reading the comand line arg
    argreader.getRemainingArgs()

    #Getting the args from command line and config file
    args = argreader.args

    con,cur = get_db(args.exp_id)
    is_multi_step_dic,const_dic = get_sal_metric_dics()
    #metric_list = list(const_dic.keys())
    #score_file_paths = get_score_file_paths(args.exp_id,metric_list)

    background_func = get_background_func(args.background)
    rng = np.random.default_rng(0)

    for metric_name in ["DAUC","IAUC"]:
        print(metric_name)

        background = background_func(metric_name)

        for cumulative_suff in ["","-nc"]:
            metric_name += cumulative_suff

            query = f'SELECT post_hoc_method,saliency_scores,metric_value FROM metrics WHERE model_id=="{args.model_id}" and metric_label=="{metric_name}" and replace_method=="{background}"'

            output = cur.execute(query).fetchall() 

            #print(query)

            output = filter_query_result(output,[0,1,2])
            
            if len(output) > 0:
                post_hoc_methods,saliency_scores_list,metric_values_list = zip(*output)

                metric_values_matrix = fmt_metric_values(metric_values_list)

                #output_list = fmt_value_str(output_list[0])
                #target_list = fmt_value_str(target_list[0])
                #inds = fmt_value_str(inds[0])
                
                #saliency_scores_list = fmt_value_str(saliency_scores_list[0])

                err_list = []
                confinterv_err_list = []

                base_err_list = []
                base_confinterv_err_list = []

                for j in range(len(post_hoc_methods)):

                    target = metric_values_matrix[:,j]
                    sal_maps = fmt_value_str(saliency_scores_list[j])
                    sal_maps = sal_maps.reshape(sal_maps.shape[0],-1)
                    n = sal_maps.shape[0]

                    #model = MLPRegressor(hidden_layer_sizes=(100,100))
                    model = svm.SVR()

                    random_maps = np.random.rand(*sal_maps.shape)

                    for maps,is_baseline in [(random_maps,True),(sal_maps,False)]:
                        mean,low,high = train_eval(maps,target,n,model,rng)

                        if is_baseline:
                            base_err_list.append(mean)
                            base_confinterv_err_list.append((low,high))
                        else:
                            err_list.append(mean)
                            confinterv_err_list.append((low,high))

                plt.figure()
                x = np.arange(len(post_hoc_methods))

                width = 0.4

                for xshift,label,mean_list,confinterv_list in [[0,"Random maps",base_err_list,base_confinterv_err_list],[width,"Real maps",err_list,confinterv_err_list]]:

                    plt.bar(x+xshift,mean_list,width=width,label=label)
                    confinterv_list = np.array(confinterv_list).transpose(1,0)
                    plt.errorbar(x+xshift,mean_list,confinterv_list,color="black",marker="o",linestyle="")

                plt.legend()
                plt.ylabel("Relative error")
                plt.xticks(x+width/2,post_hoc_methods,rotation=45,ha="right")
                plt.tight_layout()
                plt.savefig(f"../vis/{args.exp_id}/predict_{metric_name}_from_sal_maps_{args.model_id}.png")
                plt.close()

        else:
            print("RIEN CHEH")
if __name__ == "__main__":
    main()