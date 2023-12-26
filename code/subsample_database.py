import os
import argparse

import numpy as np
import pandas as pd

#Inspiration: https://doi.org/10.1093/humrep/deac171
def get_global_quality(values):

    te,icm,exp = values["TE"],values["ICM"],values["EXP"]

    if int(exp)<3 or icm=="C" or te=="C":
        quality = "low"
    elif icm=="B" or icm =="B":
        quality = "fair"
    else:
        quality = "high"

    return quality

def convert_to_cat(confidence,low_thres,high_thres):

    if confidence < low_thres:
        confidence = 0
    elif confidence < high_thres:
        confidence = 1
    else:
        confidence = 2
    
    return confidence

def get_global_confidence(values,low_thres=0.5,high_thres=0.95):

    te,icm,exp = values["TE"],values["ICM"],values["EXP"]

    te = convert_to_cat(te,low_thres,high_thres)
    icm = convert_to_cat(icm,low_thres,high_thres)
    exp = convert_to_cat(exp,low_thres,high_thres)

    total = te+icm+exp

    if total<=1:
        confidence = "low"
    elif total>=5:
        confidence = "high"
    else:
        confidence = "fair"

    return confidence

def merge_dict(img_names,aggr_annot_dict,conf_annot_dict):
    merged_dict = {}

    for image_name in img_names:
        merged_dict[image_name] = aggr_annot_dict[image_name]+"+"+conf_annot_dict[image_name]
    return merged_dict

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_fold_path",type=str)
    parser.add_argument("--aggr_annot_path",type=str)
    parser.add_argument("--conf_annot_path",type=str)
    parser.add_argument("--out_fold_path",type=str)   
    parser.add_argument("--sampling_prop",type=float,default=0.1)       
    args = parser.parse_args()

    aggr_annot_csv = pd.read_csv(args.aggr_annot_path,delimiter=" ")
    aggr_annot_dict = {row["image_name"]:get_global_quality(row) for _,row in aggr_annot_csv.iterrows()}
    
    conf_annot_csv = pd.read_csv(args.conf_annot_path,delimiter=" ")
    conf_annot_dict = {row["image_name"]:get_global_confidence(row) for _,row in conf_annot_csv.iterrows()}

    img_list = np.array(list(aggr_annot_csv["image_name"]))
    global_dict = merge_dict(img_list,aggr_annot_dict,conf_annot_dict)

    values = np.array([global_dict[image_name] for image_name in img_list])

    possible_values = list(set(global_dict.values()))

    np.random.seed(0)

    all_selected_imgs = []
    for value in possible_values:

        matching_images = img_list[values==value]

        np.random.shuffle(matching_images)

        selected_images = matching_images[:int(len(matching_images)*args.sampling_prop)]

        all_selected_imgs.extend(selected_images)

    out_path = os.path.join(args.out_fold_path,"subsample.csv")

    np.savetxt(out_path,all_selected_imgs,fmt="%s")

if __name__ == "__main__":
    main()