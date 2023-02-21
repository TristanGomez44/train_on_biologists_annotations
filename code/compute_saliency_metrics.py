
from args import ArgReader
import os
import glob
import numpy as np

from metrics import get_sal_metric_dics,add_validity_rate_multi_step,add_validity_rate_single_step
from saliency_maps_metrics.multi_step_metrics import compute_auc_metric

def fix_type(tensor):

    if type (tensor) is not np.ndarray:
        tensor = tensor.numpy()

    return tensor

def get_score_file_paths(exp_id,metric_list):
    paths = []
    for metric in metric_list:
        paths.extend(glob.glob(f"../results/{exp_id}/{metric}*_*.npy"))
    return paths

def get_col_index_to_value_dic():
    return {0:"metric_label",1:"replace_method",2:"model_id",3:"post_hoc_method",4:"metric_value"}

def write_csv(**kwargs):

    col_index_to_value_dic = get_col_index_to_value_dic()

    inds = sorted(list(col_index_to_value_dic.keys()))
    value_list = [str(kwargs[col_index_to_value_dic[ind]]) for ind in inds]
    row = ",".join(value_list)
    exp_id = kwargs["exp_id"]

    csv_path = f"../results/{exp_id}/saliency_metrics.csv"

    if not os.path.exists(csv_path):
        column_names = [col_index_to_value_dic[ind] for ind in inds]
        header = ",".join(column_names)  
        with open(csv_path,"a") as file:
            print(header,file=file)

    with open(csv_path,"a") as file:
        print(row,file=file)

def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    #Reading the comand line arg
    argreader.getRemainingArgs()

    #Getting the args from command line and config file
    args = argreader.args

    is_multi_step_dic,const_dic = get_sal_metric_dics()
    metric_list = list(const_dic.keys())
    score_file_paths = get_score_file_paths(args.exp_id,metric_list)

    for path in score_file_paths:
        
        filename = os.path.basename(path).replace(".npy","")
        
        metric_name_and_replace_method,model_id_and_posthoc_method = filename.split("_")
        
        metric_name,replace_method = metric_name_and_replace_method.split("-")
        
        if "-" in model_id_and_posthoc_method:
            model_id,post_hoc_method = model_id_and_posthoc_method.split("-")
        else:
            model_id = model_id_and_posthoc_method
            post_hoc_method = ""

        metric = const_dic[metric_name]()

        result_dic = np.load(path,allow_pickle=True).item()
        if is_multi_step_dic[metric_name]:
            all_score_list,all_sal_score_list = result_dic["prediction_scores"],result_dic["saliency_scores"]
            all_score_list = fix_type(all_score_list)
            
            auc_metric = compute_auc_metric(all_score_list)
            calibration_metric = metric.compute_calibration_metric(all_score_list, all_sal_score_list)
            result_dic = metric.make_result_dic(auc_metric,calibration_metric)

            val_rate = add_validity_rate_multi_step(metric_name,all_score_list)
            result_dic[metric_name+"_val_rate"] = val_rate

        else:
            all_score_list,all_score_masked_list = result_dic["prediction_scores"],result_dic["prediction_scores_with_mask"]
            all_score_list = fix_type(all_score_list)

            all_score_masked_list = fix_type(all_score_masked_list)
            result_dic = metric.compute_metric(all_score_list,all_score_masked_list)
            
            val_rate = add_validity_rate_single_step(metric_name,all_score_list,all_score_masked_list)
            result_dic[metric_name+"_val_rate"] = val_rate

        for sub_metric in result_dic.keys():
            write_csv(exp_id=args.exp_id,metric_label=sub_metric.upper(),replace_method=replace_method,model_id=model_id,post_hoc_method=post_hoc_method,metric_value=result_dic[sub_metric])

if __name__ == "__main__":
    main()
