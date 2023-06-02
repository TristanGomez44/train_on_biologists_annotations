import modelBuilder,load_data,trainVal
from args import ArgReader
import ssl_dataset
from load_data import get_img_size
import os,sys,shutil
import subprocess
import numpy as np
import matplotlib.pyplot as plt 

def main(argv=None):
    # Getting arguments from config file and command line
    # Building the arg reader
    argreader = ArgReader(argv)

    argreader = modelBuilder.addArgs(argreader)
    argreader = load_data.addArgs(argreader)
    argreader = trainVal.addSSLArgs(argreader)

    argreader.parser.add_argument('--layers_to_show', type=str, nargs="*")
    argreader.parser.add_argument('--model_ids', type=str, nargs="*")
    argreader.parser.add_argument('--viz_id', type=str,default="")
    argreader.parser.add_argument('--log', action='store_true')

    # Reading the comand line arg
    argreader.getRemainingArgs()

    args = argreader.args

    if args.layers_to_show is None:
        layers_to_show = ["module.firstModel.featMod.layers.0.blocks.1.mlp.fc2.weight",\
                          "module.firstModel.featMod.layers.1.blocks.1.mlp.fc2.weight",\
                          "module.firstModel.featMod.layers.2.blocks.17.mlp.fc2.weight",\
                          "module.firstModel.featMod.layers.3.blocks.1.mlp.fc2.weight",\
                          "module.secondModel.lin_lay_exp.weight",\
                          "module.secondModel.lin_lay_icm.weight",\
                          "module.secondModel.lin_lay_te.weight"]
    else:
        layers_to_show = args.layers_to_show

    all_csv_dic = {}
    for model_id in args.model_ids:
        all_csv_dic[model_id] = {}
        for stat in ["mean","std"]:
            all_csv_dic[model_id][stat] = np.genfromtxt(f"../results/{args.exp_id}/gradnorm_{stat}_{model_id}.csv",delimiter=",",dtype=str)

    #plt.figure(figsize=(30,10))
    #subplots 
    fig, axs = plt.subplots(len(layers_to_show),1,figsize=(30,10))

    for i,layer in enumerate(layers_to_show):
        
        for model_id in args.model_ids:

            csv_dic = all_csv_dic[model_id]

            col_ind_in_csv = np.where(csv_dic["mean"][0]==layer)[0][0]

            csv_mean = csv_dic["mean"]
            csv_std = csv_dic["std"]
            epochs = csv_mean[1:,0].astype(float)
            batch_indexes = csv_mean[1:,1].astype(float)

            step_nb_per_epoch = max(batch_indexes)

            global_steps = (epochs-1)*step_nb_per_epoch+batch_indexes

            means = csv_mean[1:,col_ind_in_csv].astype(float)
            stds = csv_std[1:,col_ind_in_csv].astype(float)

            #Plot mean and stds 
            axs[i].plot(global_steps,means,label=model_id)
            axs[i].set_title(layer)
            if args.log:
                axs[i].set_yscale('log')
            else:
                axs[i].fill_between(global_steps,means-stds,means+stds,alpha=0.2)
        
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"../vis/{args.exp_id}/gradnorm_{args.viz_id}_log={args.log}.png")

if __name__ == "__main__":
    main()