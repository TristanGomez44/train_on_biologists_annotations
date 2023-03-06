
from args import ArgReader
import os
import glob
import numpy as np
import sqlite3 
def make_dic(csv):

    dic = {}

    for row in csv[1:]:

        key = " ".join(row[:-1])
        dic[key] = row[-1]
        print(key)    
    return dic

def csv_from_db(db_path):
    con = sqlite3.connect(db_path) # change to 'sqlite:///your_filename.db'
    cur = con.cursor()
    csv = cur.execute(f"SELECT * FROM metrics;") # use your column names here
    header = [description[0] for description in csv.description]
    csv = csv.fetchall()
    csv = [header] + csv
    csv = np.array(csv,dtype=str)

    return csv

def write_table(metrics,model_ids,background_func,post_hoc_method,dic,path):

    table = 'model_id,' + ",".join(metrics) + "\n"
    for model_id in model_ids:

        row = model_id+","

        value_list = []
        for metric in metrics:
            background = background_func(metric)
            key = " ".join([metric,background,model_id,post_hoc_method])

            value = dic[key]

            value_list.append(value)
        
        row += ",".join(value_list) + "\n"

        table += row 

    with open(path,"w") as file:
        print(table,file=file)
  

def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    #Reading the comand line arg
    argreader.getRemainingArgs()

    #Getting the args from command line and config file
    args = argreader.args

    exp_id = "CROHN25"
    model_ids = ["noneRed2","noneRed_onlyfocal","noneRed_onlylossonmasked","noneRed_focal2"]
    metrics = ["DAUC","DC","IAUC","IC","IIC","AD","ADD"]
    background_func = lambda x:"blur" if x in ["IAUC","IC","INSERTION_VAL_RATE"] else "black"

    post_hoc_method = "gradcampp"

    csv = csv_from_db(f"../results/CROHN25/saliency_metrics.db")

    dic = make_dic(csv)

    path = f"../results/{exp_id}/does_calibration_improve_faithfulness.csv"
    write_table(metrics,model_ids,background_func,post_hoc_method,dic,path)

    metrics = ["IIC_AD_VAL_RATE","ADD_VAL_RATE","INSERTION_VAL_RATE","DELETION_VAL_RATE"]
    path = f"../results/{exp_id}/does_calibration_improve_faithfulness_val_rate.csv"
    write_table(metrics,model_ids,background_func,post_hoc_method,dic,path)
            
if __name__ == "__main__":
    main()
