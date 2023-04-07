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

    no_calibration_csv = np.genfromtxt(f"../results/{args.exp_id}/krippendorff_alpha_noneRed2_bNone.csv",dtype=str,
    
    delimiter=":")
    calibration_csv = np.genfromtxt(f"../results/{args.exp_id}/krippendorff_alpha_noneRed_focal2_bNone.csv",dtype=str,delimiter=":")

    no_cal_cumulative = no_calibration_csv[1].split(",")[1]
    cal_cumulative = calibration_csv[1].split(",")[1]
    cal_non_cumulative = calibration_csv[2].split(",")[1]

    plt.figure()
    plt.rc('axes', axisbelow=True)
    plt.grid(linestyle='--')

    values = [no_cal_cumulative,cal_cumulative,cal_non_cumulative]

    fontsize = 17

    for i,value in enumerate(values):

        value = value.replace("(","").replace(")","")
        mean,low,high = value.split(" ")
        mean,low,high = float(mean),float(low),float(high)
        low,high = mean-low,high-mean

        plt.bar(i,mean,edgecolor="black")
        plt.errorbar(i,mean,np.array([low,high])[:,np.newaxis],fmt='none',color="black")
    
    plt.ylabel("Metrics reliability",fontsize=fontsize)

    locs = [0,1,2,2.0001]
    labels = ["Baseline","Calibrated","Calibrated+","\nNon-cumulative"]

    plt.xticks(locs,labels,rotation=0,ha="center",fontsize=fontsize)

    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    plt.savefig(f"../vis/{args.exp_id}/teaser_bar_plot.png")
    plt.close()
    
if __name__ == "__main__":
    main()