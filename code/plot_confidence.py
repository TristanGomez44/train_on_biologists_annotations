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
    argreader = addInitArgs(argreader)
    argreader = addValArgs(argreader)
    argreader = addLossTermArgs(argreader)
    argreader = init_post_hoc_arg(argreader)
    argreader = addSalMetrArgs(argreader)
    argreader = modelBuilderArgs(argreader)
    argreader = loadDataArgs(argreader)

    argreader.parser.add_argument('--n_bins',type=int,default=15)
  
    # Reading the comand line arg
    argreader.getRemainingArgs()

    args = argreader.args

    args.cuda = args.cuda and torch.cuda.is_available()

    if args.class_nb is None:
        args.class_nb = get_class_nb(args.dataset_train)

    res_path = f"../results/{args.exp_id}/conf_and_acc_list_{args.model_id}.npy"

    if not os.path.exists(res_path):

        valLoader,_ = buildTestLoader(args,"test")

        net = netBuilder(args)
        initialize_Net_And_EpochNumber(net, args.exp_id, args.model_id, args.cuda, "fine_tune",args.init_path,False)

        args.compute_masked = True
        ece,conf_list,acc_list = compute_ece_hist_on_test_set(net, valLoader, args, mode="val",n_bins=args.n_bins)
        res_dict = {"conf_list":conf_list,"acc_list":acc_list,"ece":ece}
        np.save(res_path,res_dict)
    else:
        res_dict = np.load(res_path,allow_pickle=True).item()
        conf_list,acc_list,ece = res_dict["conf_list"],res_dict["acc_list"],res_dict["ece"]

    conf_list = (conf_list[:-1] + conf_list[1:]) * 0.5

    conf_list_and_acc_list = np.stack((conf_list,acc_list),axis=-1)
    conf_list_and_acc_list = list(filter(lambda x:x[1] != 0,conf_list_and_acc_list))
    conf_list_and_acc_list = np.array(conf_list_and_acc_list)
    conf_list,acc_list = conf_list_and_acc_list[:,0],conf_list_and_acc_list[:,1]

    plt.figure()
    plt.rc('axes', axisbelow=True)
    plt.grid(linestyle='--')
    width = conf_list[1]-conf_list[0]

    diff = np.abs(acc_list-conf_list)
    underconf = -np.minimum(conf_list-acc_list,0)
    overconf = np.minimum(acc_list-conf_list,0)
    error = np.minimum(acc_list,conf_list)

    fontsize = 17
    fig, ax = plt.subplots(nrows=1, ncols=1)
    plt.bar(conf_list,error,edgecolor="black",color="darkblue",width=width,alpha=0.75)
    plt.bar(conf_list,underconf,bottom=conf_list,edgecolor="black",color="firebrick",width=width,alpha=0.75,hatch="xxx")
    plt.bar(conf_list,overconf,bottom=conf_list,edgecolor="black",color="firebrick",width=width,alpha=0.75,hatch="...")
    plt.plot([0,1],[0,1],"--",color="gray",linewidth=4)
    
    x,y = 0.66,0.035
    
    width = 1-x-0.001
    plt.text(x,y,"ECE="+str(round(ece*100,2)),fontsize=int(fontsize*1.4))
    #rect = patches.Rectangle((x,y-0.009), width, 0.1, linewidth=1, edgecolor="none", facecolor="white",alpha=0.75)        
    
    rect = patches.FancyBboxPatch((x-0.006,y-0.014),width, 0.1,boxstyle="round,pad=0.0040,rounding_size=0.015",ec="none", fc="white",mutation_aspect=0.5,alpha=0.7)
    
    ax.add_patch(rect)

    plt.ylabel("Accuracy",fontsize=fontsize)
    plt.xlabel("Confidence",fontsize=fontsize)
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.tight_layout()
    
    plt.savefig(f"../vis/{args.exp_id}/confidence_vs_accuracy_{args.model_id}.png")
    plt.close()

def compute_ece_hist_on_test_set(model, loader, args, mode="val",n_bins=15):

    model.eval()

    var_dic = {}
    for batch_idx, batch in enumerate(loader):
        data, target = batch[:2]

        if (batch_idx % args.log_interval == 0):
            print("\t", batch_idx * len(data), "/", len(loader.dataset))

        # Puting tensors on cuda
        if args.cuda: data, target = data.cuda(non_blocking=True), target.cuda(non_blocking=True)

        resDict = {}
        
        other_data = batch[2].to(data.device) if args.sal_metr_otherimg else None
        resDict,data,_= apply_sal_metr_masks_and_update_dic(model,data,args,resDict,other_data)
                        
        resDict.update(model(data))

        var_dic = all_cat_var_dic(var_dic,resDict,target,args,mode)

    ece_values,conf_list,acc_list = ece(var_dic["output"], var_dic["target"],n_bins=n_bins,return_conf_and_acc=True)

    return ece_values,np.array(conf_list),np.array(acc_list)

if __name__ == "__main__":
    main()
