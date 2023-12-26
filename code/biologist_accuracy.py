import os
import argparse

import numpy as np
import pandas as pd

import json
import matplotlib.pyplot as plt

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--split_file_path",type=str)
    parser.add_argument("--conf_annot_path",type=str)
    parser.add_argument("--out_fold_path",type=str)   
    args = parser.parse_args()

    conf_annot_csv = pd.read_csv(args.conf_annot_path,delimiter=" ")
    conf_annot_dict = {}

    with open(args.split_file_path) as f:
        test_images = json.load(f)["test"]
    
    keys = ["TE","ICM","EXP"]
    #fig,axs = plt.subplots(nrows=len(keys),nrows=1)

    for i,key in enumerate(keys):
        conf_annot_dict = {row["image_name"]:row[key] for _,row in conf_annot_csv.iterrows()}

        perf = []
        for img_name in test_images:
            perf.append(conf_annot_dict[img_name])
        perf = np.array(perf)

        print(key,perf.mean())

if __name__ == "__main__":
    main()