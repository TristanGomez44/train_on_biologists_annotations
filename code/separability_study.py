
import os ,sys

import numpy as np 

from args import addInitArgs,addValArgs
import modelBuilder,load_data
from args import ArgReader

def list_to_str(values):
    string = str(round(values.mean(),4)) + "±" +str(round(values.std(),4))
    return string

def main(argv=None):
    # Getting arguments from config file and command line
    # Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--transf1', type=str,default="identity")
    argreader.parser.add_argument('--transf2', type=str,default="identity")
    argreader.parser.add_argument('--model', type=str,default="SVM")
    argreader.parser.add_argument('--normalize', action="store_true")
    argreader.parser.add_argument('--folds', type=int,default=10)
    argreader.parser.add_argument('--model_id2', type=str)

    argreader = addInitArgs(argreader)
    argreader = addValArgs(argreader)

    argreader = modelBuilder.addArgs(argreader)
    argreader = load_data.addArgs(argreader)

    # Reading the comand line arg
    argreader.getRemainingArgs()

    args = argreader.args

    print("Model id",args.model_id,"Transf",args.transf1,args.transf2)

    if args.model_id2 is None:
        args.model_id2 = args.model_id

    repres1 = np.load(f"../results/{args.exp_id}/img_repr_model{args.model_id}_transf{args.transf1}.npy")
    repres2 = np.load(f"../results/{args.exp_id}/img_repr_model{args.model_id2}_transf{args.transf2}.npy")

    sep_dict = run_separability_analysis(repres1,repres2,args.normalize,args.seed,args.folds)

    train_acc,train_auc = sep_dict["train_acc"],sep_dict["train_auc"]
    val_acc,val_auc = sep_dict["val_acc"],sep_dict["val_auc"]

    train_inv_auc,val_inv_auc = sep_dict["train_inv_auc"],sep_dict["val_inv_auc"]

    train_acc = list_to_str(train_acc)
    train_auc = list_to_str(train_auc)
    val_acc = list_to_str(val_acc)
    val_auc = list_to_str(val_auc)

    #train_inv_auc = list_to_str(train_inv_auc)
    #val_inv_auc = list_to_str(val_inv_auc)

    csv_path = f"../results/{args.exp_id}/separability_study.csv"
    if not os.path.exists(csv_path):
        with open(csv_path,"w") as file:
            print(f"model_id,tranfs1,transf2,sec_model,train_acc,train_auc,val_acc,val_auc",file=file) 

    sec_model = args.model 
    if args.normalize:
        sec_model += "_norm"

    if args.model_id == args.model_id2:
        model_ids = args.model_id 
    else:
        model_ids = args.model_id + "_and_" + args.model_id2

    with open(csv_path,"a") as file:
        print(f"{model_ids},{args.transf1},{args.transf2},{sec_model},{train_acc},{train_auc},{val_acc},{val_auc}",file=file)

if __name__ == "__main__":
    main()