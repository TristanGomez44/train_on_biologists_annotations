
from args import ArgReader
import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from compute_scores_for_saliency_metrics import get_metric_dics
from saliency_maps_metrics.multi_step_metrics import compute_auc_metric

def get_score_file_paths(exp_id,metric_list):
    paths = []
    for metric in metric_list:
        paths.extend(glob.glob(f"../results/{exp_id}/{metric}*_*.npy"))
    return paths

def write_csv(mean,metric_name_and_replace_method,exp_id,model_id):
    with open(f"../results/{exp_id}/saliency_metrics.csv","a") as file:
        print(f"{model_id},{metric_name_and_replace_method},{mean}",file=file)

def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    #Reading the comand line arg
    argreader.getRemainingArgs()

    #Getting the args from command line and config file
    args = argreader.args

    is_multi_step_dic,const_dic = get_metric_dics()
    metric_list = list(const_dic.keys())
    score_file_paths = get_score_file_paths(args.exp_id,metric_list)

    for path in score_file_paths:
        
        filename = os.path.basename(path)
        
        model_id = filename.split("_")[-1].replace(".npy","")
        metric_name_and_replace_method = "_".join(filename.split("_")[:-1])
        
        if "-" in metric_name_and_replace_method:
            metric_name,_ = metric_name_and_replace_method.split("-")
        else:
            metric_name = metric_name_and_replace_method

        metric = const_dic[metric_name]()

        result_dic = np.load(path,allow_pickle=True).item()
        if is_multi_step_dic[metric_name]:
            all_score_list,all_sal_score_list = result_dic["prediction_scores"],result_dic["saliency_scores"]
            
            mean_auc = compute_auc_metric(all_score_list)
            mean_calibration = metric.compute_calibration_metric(all_score_list, all_sal_score_list)

            write_csv(mean_auc,metric_name_and_replace_method+"-auc",args.exp_id,model_id)
            write_csv(mean_calibration,metric_name_and_replace_method+"-cal",args.exp_id,model_id)
            
        else:
            all_score_list,all_score_masked_list = result_dic["prediction_scores"],result_dic["prediction_scores_with_mask"]
            result_dic = metric.compute_metric(all_score_list,all_score_masked_list)

            for key in result_dic.keys():
                write_csv(result_dic[key],metric_name_and_replace_method+"-"+key,args.exp_id,model_id)

if __name__ == "__main__":
    main()
