
from args import ArgReader
import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import pearsonr
from does_calibration_improve_faithfulness import csv_from_db

def make_temp_to_faithmetric_dict(path,ref_model_id,background_func,ref_post_hoc_method):

    csv = csv_from_db(path)
    
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
    model_id = "noneRed_focal2"
    background_func = lambda x:"blur" if x in ["IAUC","IC","INSERTION_VAL_RATE"] else "black"

    post_hoc_method = "gradcampp"

    temp_to_faith = make_temp_to_faithmetric_dict(f"../results/CROHN25/saliency_metrics.db",model_id,background_func,post_hoc_method)

    ece_to_temp = make_ece_to_temp_dict("../results/CROHN25/metrics_noneRed_focal2_test.csv")

    #for key in ece_to_temp:
    #    print(key,ece_to_temp[key],type(ece_to_temp[key]))
    #print("Temp")
    #for key in temp_to_faith["DAUC"]:
    #    print(type(key),key,temp_to_faith["DAUC"][key])

    cmap = plt.get_cmap("rainbow")

    for metric in temp_to_faith:
        print(metric)
        ece_list=[]
        metric_value_list=[]
        temp_list = []
        for ece in ece_to_temp:

            temp = ece_to_temp[ece]
            metric_value = temp_to_faith[metric][temp]

            ece_list.append(ece)
            metric_value_list.append(metric_value)
        
            temp_list.append(temp)

        correlation,pvalue = pearsonr(ece_list,metric_value_list)

        print(correlation,pvalue)

        temp_list = np.array(temp_list)

        plt.figure()
        plt.scatter(ece_list,metric_value_list,color=cmap(temp_list/max(temp_list)))
        plt.ylabel(metric)
        plt.xlabel("ece")
        plt.savefig(f"../vis/{exp_id}/{metric}_vs_ece.png")
            
if __name__ == "__main__":
    main()
