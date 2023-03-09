
from args import ArgReader
import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import pearsonr

import sqlite3 ,csv as csvLib,os,sys

def get_metric_name(is_corr_metric,is_deletion_metric):

    if is_corr_metric:
        if is_deletion_metric:
            metric_name = "DC"
        else:
            metric_name = "IC"
    else:
        if is_deletion_metric:
            metric_name = "DAUC"
        else:
            metric_name = "IAUC"
    
    return metric_name

def make_temp_to_faithmetric_dict(path,ref_model_id,background_func,ref_post_hoc_method):

    csv = np.genfromtxt(path,delimiter=",",dtype=str)
    
    temp_to_faith_dict = {}
    for row in csv[1:]:

        metric,background,model_id,post_hoc_method,metric_value = row

        if not metric in temp_to_faith_dict:
            temp_to_faith_dict[metric] = {}

        if ref_post_hoc_method == post_hoc_method and background_func(metric) == background and ref_model_id in model_id:

            if "T" in model_id:
                model_id,temp = model_id.split("T")
                temp = float(temp)
            else:
                temp = 1

            if "\pm" in metric_value:
                metric_value = metric_value.split("\pm")[0]

            temp_to_faith_dict[metric][temp] = float(metric_value)
        
    return temp_to_faith_dict

def make_ece_to_temp_dict(path):

    csv = np.genfromtxt(path,delimiter=",",dtype=str)

    header = csv[0]

    ece_col = np.where(header=="ece")[0][0]
    temp_col = np.where(header=="temperature")[0][0]

    ece_values= csv[1:,ece_col].astype("float")
    temp_values = csv[1:,temp_col]

    ece_to_temp_dict = {ece:float(temp) for ece,temp in zip(ece_values,temp_values)}

    return ece_to_temp_dict

def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    #Reading the comand line arg
    argreader.getRemainingArgs()

    #Getting the args from command line and config file
    args = argreader.args

    exp_id = "CROHN25"
    model_ids = ["noneRed2","noneRed_focal2"]
    background_func = lambda x:"blur" if x in ["IAUC","IC","INSERTION_VAL_RATE"] else "black"
    post_hoc_method = "gradcampp"
    db_path = f"../results/{exp_id}/saliency_metrics.db"
    
    con = sqlite3.connect(db_path) # change to 'sqlite:///your_filename.db'
    cur = con.cursor()

    csv =""

    for model_id in model_ids:

        csv += model_id+",Deletion,Insertion\n"

        for is_corr_metric in [True,False]:

            if is_corr_metric:
                row = "Corr"
            else:
                row = "AuC"

            for is_deletion_metric in [True,False]:

                metric_label = get_metric_name(is_corr_metric,is_deletion_metric)
                background = background_func(metric_label)
    
                metric_value = cur.execute(f'SELECT metric_value FROM metrics WHERE post_hoc_method=="{post_hoc_method}" and model_id=="{model_id}" and metric_label=="{metric_label}" and replace_method=="{background}"').fetchall()[0][0]

                metric_var = str(np.array(metric_value.split(";")).astype("float").std())

                row += ","+metric_var
                
            row += "\n"

            csv += row

    with open(f"../results/{exp_id}/does_auc_has_higher_var_than_corr.csv","w") as file:
        print(csv,file=file)

            
if __name__ == "__main__":
    main()
