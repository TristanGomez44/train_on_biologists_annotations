
from args import ArgReader
import os,sys
import glob
import numpy as np
import sqlite3 
from metrics import krippendorff_alpha
def fmt_metric_values(metric_values_list):

    matrix = []

    for i in range(len(metric_values_list)):
        matrix.append(np.array(metric_values_list[i].split(";")).astype("float"))

    metric_values_matrix = np.stack(matrix,axis=0)

    print(metric_values_matrix.shape)

    return metric_values_matrix

def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    #Reading the comand line arg
    argreader.getRemainingArgs()

    #Getting the args from command line and config file
    args = argreader.args

    exp_id = "CROHN25"
    #model_ids = ["noneRed2","noneRed_onlyfocal","noneRed_onlylossonmasked","noneRed_focal2"]
    model_id = "noneRed_focal2"
    metrics = ["DAUC","DC","IAUC","IC"]
    background_func = lambda x:"blur" if x in ["IAUC","IC","INSERTION_VAL_RATE"] else "black"

    db_path = f"../results/{exp_id}/saliency_metrics.db"
   
    con = sqlite3.connect(db_path) # change to 'sqlite:///your_filename.db'
    cur = con.cursor()

    csv = "cumulative,"+ ",".join(metrics) + "\n"
    
    post_hoc_methods_list = []

    for cumulative_suff in ["","-nc"]:

        csv += "True" if cumulative_suff == "-nc" else "False"

        for metric in metrics:
            background = background_func(metric)
            metric += cumulative_suff
            output = cur.execute(f'SELECT post_hoc_method,metric_value FROM metrics WHERE model_id=="{model_id}" and metric_label=="{metric}" and replace_method=="{background}"').fetchall()

            post_hoc_methods,metric_values_list = zip(*output)

            metric_values_matrix = fmt_metric_values(metric_values_list)

            alpha = krippendorff_alpha(metric_values_matrix)

            csv += ","+str(alpha)

            post_hoc_methods_list.append(post_hoc_methods)

        csv += "\n"

    with open(f"../results/{exp_id}/krippendorff_alpha.csv","w") as file:
        print(csv,file=file)

    post_hoc_methods_set = set(post_hoc_methods_list)

    if len(post_hoc_methods_set) != -1:
        print("Different sets of posthoc methods were used:",post_hoc_methods_set)

if __name__ == "__main__":
    main()
