import os

import torch 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from args import ArgReader,str2bool,addInitArgs,addValArgs,init_post_hoc_arg,addLossTermArgs,addSalMetrArgs
from modelBuilder import addArgs as modelBuilderArgs, netBuilder
from load_data import addArgs as loadDataArgs, get_class_nb,buildTestLoader
from init_model import initialize_Net_And_EpochNumber
from update import all_cat_var_dic
from sal_metr_data_aug import apply_sal_metr_masks_and_update_dic
from metrics import ece

def main(argv=None):
    # Getting arguments from config file and command line
    # Building the arg reader
    argreader = ArgReader(argv)
    # Reading the comand line arg
    argreader.getRemainingArgs()

    args = argreader.args

    no_calibration_csv = np.genfromtxt(f"../results/{args.exp_id}/krippendorff_alpha_noneRed2_bNone.csv",dtype=str,delimiter=":")

    header = np.array(no_calibration_csv[0].split(","))

    
    calibration_csv = np.genfromtxt(f"../results/{args.exp_id}/krippendorff_alpha_noneRed_focal2_bNone.csv",dtype=str,delimiter=":")

    plt.figure(figsize=(7,5))
    plt.rc('axes', axisbelow=True)
    plt.grid(linestyle='--')
    fontsize = 29

    colors = ["blue","orange","green"]

    width = 0.2

    for i,metric in enumerate(["DAUC","IAUC"]):

        col_ind = np.argwhere(header == metric)[0][0]

        no_cal_cumulative = no_calibration_csv[1].split(",")[col_ind]
        cal_cumulative = calibration_csv[1].split(",")[col_ind]
        cal_non_cumulative = calibration_csv[2].split(",")[col_ind]

        values = [no_cal_cumulative,cal_cumulative,cal_non_cumulative]
        labels = ["Baseline","Calibrated","Calibrated+Non-cumulative"]

        for j,(value,label,color) in enumerate(zip(values,labels,colors)):

            value = value.replace("(","").replace(")","")
            mean,low,high = value.split(" ")
            mean,low,high = float(mean),float(low),float(high)
            low,high = mean-low,high-mean

            xpos = i+j/5-1/5
            if i == 0:
                plt.bar(xpos,mean,width=width,edgecolor="black",label=label,color=color)
            else:
                plt.bar(xpos,mean,width=width,edgecolor="black",color=color)
            plt.errorbar(xpos,mean,np.array([low,high])[:,np.newaxis],fmt='none',color="black")
    
    plt.ylabel("Metrics reliability",fontsize=fontsize)
    plt.legend(prop={'size': int(fontsize*0.65)})
    #locs = [0,1,2,2.0001]
    #labels = ["Baseline","Calibrated","Calibrated+","\nNon-cumulative"]
    #plt.xticks(locs,labels,rotation=0,ha="center",fontsize=int(fontsize*0.7))

    plt.xticks([0,1],["Deletion","Insertion"],fontsize=fontsize)

    plt.yticks(np.arange(7)/10,fontsize=int(fontsize*0.85))
    plt.tight_layout()
    plt.savefig(f"../vis/{args.exp_id}/teaser_bar_plot.png")
    plt.close()
    
if __name__ == "__main__":
    main()