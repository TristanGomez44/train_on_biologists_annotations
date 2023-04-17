
from args import ArgReader
import os,sys
import glob
import numpy as np
import torch 

from metrics import get_sal_metric_dics,add_validity_rate_multi_step,add_validity_rate_single_step
from saliency_maps_metrics.multi_step_metrics import compute_auc_metric
from modelBuilder import addArgs as addArgsModelBuilder

import sqlite3 ,csv as csvLib,os,sys

def get_db(exp_id):
    db_path = f"../results/{exp_id}/saliency_metrics.db"
    if not os.path.exists(db_path):
        col_list = get_header()
        make_db(col_list,db_path)
    con = sqlite3.connect(db_path) # change to 'sqlite:///your_filename.db'
    cur = con.cursor()
    return con,cur 

def make_db(col_list,db_path):

    con = sqlite3.connect(db_path) # change to 'sqlite:///your_filename.db'
    cur = con.cursor()
    header = ",".join(col_list)
    cur.execute(f"CREATE TABLE metrics ({header},UNIQUE({header}) ON CONFLICT IGNORE);") # use your column names here

    con.commit()

def apply_softmax(array,temperature):
    tensor = torch.from_numpy(array)
    tensor = torch.softmax(tensor.double()/temperature,dim=-1)

    if len(tensor.shape) == 3:
        inds = tensor[:,0].argmax(dim=-1,keepdim=True).unsqueeze(1)
        inds = inds.expand(-1,tensor.shape[1],-1)
        tensor = tensor.gather(2,inds).squeeze(-1)
    else:
        inds = tensor.argmax(dim=-1,keepdim=True)
        tensor = tensor.gather(1,inds).squeeze(-1)        

    array = tensor.numpy()
    return array

def fix_type(tensor):

    if type (tensor) is not np.ndarray:
        tensor = tensor.numpy()

    return tensor

def get_score_file_paths(exp_id,metric_list,pref=""):
    paths = []
    for metric in metric_list:
        metric = metric.replace("_","")
        paths.extend(glob.glob(f"../results/{exp_id}/{pref}{metric}*_*.npy"))
    return paths

def get_supp_kw():
    return {5:"saliency_scores",6:"outputs",7:"target",8:"inds",9:"prediction_scores"}

def get_col_index_to_value_dic():
    dic = {0:"metric_label",1:"replace_method",2:"model_id",3:"post_hoc_method",4:"metric_value"}
    dic.update(get_supp_kw())
    return dic 

def write_db(cur,**kwargs):

    col_index_to_value_dic = get_col_index_to_value_dic()
    inds = sorted(list(col_index_to_value_dic.keys()))
    header = [col_index_to_value_dic[ind] for ind in inds]
    header = ",".join(header)

    value_list = [str(kwargs[col_index_to_value_dic[ind]]) for ind in inds]

    question_marks = ",".join(['?' for _ in range(len(inds))])
    cur.execute(f"INSERT INTO metrics ({header}) VALUES ({question_marks});", value_list)

def get_header():
    col_index_to_value_dic = get_col_index_to_value_dic()
    inds = sorted(list(col_index_to_value_dic.keys()))
    column_names = [col_index_to_value_dic[ind] for ind in inds]
    return column_names

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

def list_to_fmt_str(array):
    if type(array) is torch.Tensor:
        array = array.cpu().numpy()
    if len(array.shape) > 1:
        fmt_str = f"shape={str(array.shape)};"
    else:
        fmt_str = ""
    fmt_str += str(";".join(array.reshape(-1).astype("str")))  
    return fmt_str

def get_info(path):

    filename = os.path.basename(path).replace(".npy","")
        
    underscore_ind = filename.find("_")
    metric_name_and_replace_method,model_id_and_posthoc_method = filename[:underscore_ind],filename[underscore_ind+1:]
    
    metric_name,replace_method = metric_name_and_replace_method.split("-")
    
    if metric_name == "IICAD":
        metric_name = "IIC_AD"

    kwargs = {}
    if "nc" in metric_name:
        metric_name = metric_name.replace("nc","")
        kwargs["cumulative"] = False
        suff = "-nc"
    else:
        suff = ""

    if "-" in model_id_and_posthoc_method:
        model_id,post_hoc_method = model_id_and_posthoc_method.split("-")
    else:
        model_id = model_id_and_posthoc_method
        post_hoc_method = ""
    
    return model_id,post_hoc_method,metric_name,replace_method,kwargs,suff

def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--model_ids', type=str,nargs="*", help='Authorized model IDs. Do not set this arg to authorize all model.')

    argreader = addArgsModelBuilder(argreader)

    #Reading the comand line arg
    argreader.getRemainingArgs()

    #Getting the args from command line and config file
    args = argreader.args

    is_multi_step_dic,const_dic = get_sal_metric_dics()
    metric_list = list(const_dic.keys())
    score_file_paths = get_score_file_paths(args.exp_id,metric_list)
    
    con,cur = get_db(args.exp_id)

    for path in score_file_paths:
        
        model_id,post_hoc_method,metric_name,replace_method,kwargs,suff = get_info(path)

        if args.model_ids is None or model_id in args.model_ids:

            metric = const_dic[metric_name](**kwargs)

            scores_dic = np.load(path,allow_pickle=True).item()
            if is_multi_step_dic[metric_name]:
                all_score_list,all_sal_score_list = scores_dic["prediction_scores"],scores_dic["saliency_scores"]
                all_score_list = fix_type(all_score_list)
                
                if len(all_score_list.shape) == 3:
                    all_score_list = apply_softmax(all_score_list,args.temperature)
                    if args.temperature != 1:
                        model_id += "T"+str(args.temperature)

                auc_metric = compute_auc_metric(all_score_list)        
                calibration_metric = metric.compute_calibration_metric(all_score_list, all_sal_score_list)

                auc_metric = list_to_fmt_str(auc_metric)
                calibration_metric = list_to_fmt_str(calibration_metric)

                result_dic = metric.make_result_dic(auc_metric,calibration_metric)

                val_rate = add_validity_rate_multi_step(metric_name,all_score_list)
                result_dic[metric_name+"_val_rate"] = val_rate

            else:
                all_score_list,all_score_masked_list = scores_dic["prediction_scores"],scores_dic["prediction_scores_with_mask"]
                all_score_list = fix_type(all_score_list)
                all_score_masked_list = fix_type(all_score_masked_list)
                
                if len(all_score_list.shape) == 2:
                    all_score_list = apply_softmax(all_score_list,args.temperature)
                    all_score_masked_list = apply_softmax(all_score_masked_list,args.temperature)
                    if args.temperature != 1:
                        model_id += "T"+str(args.temperature)

                result_dic = metric.compute_metric(all_score_list,all_score_masked_list)
                
                for sub_metric in result_dic:
                    result_dic[sub_metric] = list_to_fmt_str(result_dic[sub_metric])

                val_rate = add_validity_rate_single_step(metric_name,all_score_list,all_score_masked_list)
                result_dic[metric_name+"_val_rate"] = val_rate

            supp_kwargs = {}
            for supp_kw in get_supp_kw().values():
                supp_kwargs[supp_kw] = list_to_fmt_str(scores_dic[supp_kw]) if supp_kw in scores_dic else None

            for sub_metric in result_dic.keys():
                #write_csv(exp_id=args.exp_id,metric_label=sub_metric.upper(),replace_method=replace_method,model_id=model_id,post_hoc_method=post_hoc_method,metric_value=result_dic[sub_metric])

                write_db(cur,exp_id=args.exp_id,metric_label=sub_metric.upper()+suff,replace_method=replace_method,model_id=model_id,post_hoc_method=post_hoc_method,metric_value=result_dic[sub_metric],**supp_kwargs)

    con.commit()
    con.close()

if __name__ == "__main__":
    main()
