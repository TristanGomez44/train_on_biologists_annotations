
import os ,sys

import numpy as np 
import torch
from sklearn import svm
import sklearn.metrics

from args import addInitArgs,addValArgs
import modelBuilder,load_data
from args import ArgReader

import matplotlib.pyplot as plt

def run_separability_analysis(repres1,repres2,normalize,seed,folds=10):

    len1 = len(repres1)
    len2 = len(repres2)

    if normalize:
        repres1 = repres1/np.abs(repres1).sum(axis=1,keepdims=True)
        repres2 = repres2/np.abs(repres2).sum(axis=1,keepdims=True)

    labels1 = np.zeros(len1).astype("int")
    labels2 = np.ones(len2).astype("int")

    torch.manual_seed(seed)
    np.random.seed(seed)

    train_acc = []
    train_auc = []
    val_acc = []
    val_auc = []

    train_inv_auc = []
    val_inv_auc = []   

    for i in range(folds):

        inds = np.random.permutation(len1)

        repr1_perm,lab1_perm = repres1[inds],labels1[inds]
        repr2_perm,lab2_perm = repres2[inds],labels2[inds]

        train_x = np.concatenate((repr1_perm[:len1//2],repr2_perm[:len2//2]),axis=0)
        test_x = np.concatenate((repr1_perm[len1//2:],repr2_perm[len2//2:]),axis=0)

        train_y = np.concatenate((lab1_perm[:len1//2],lab2_perm[:len2//2]),axis=0)
        test_y = np.concatenate((lab1_perm[len1//2:],lab2_perm[len2//2:]),axis=0)

        model = svm.SVC(probability=True)
        model.fit(train_x,train_y)

        train_y_score = model.predict_proba(train_x)[:,1]
        train_acc.append(model.score(train_x,train_y))
        train_auc.append(sklearn.metrics.roc_auc_score(train_y,train_y_score))
        
        test_y_score = model.predict_proba(test_x)[:,1]
        val_acc.append(model.score(test_x,test_y))
        val_auc.append(sklearn.metrics.roc_auc_score(test_y,test_y_score))

        '''
        #Inversed model
        train_inv_y_score = 1 - train_y_score
        train_inv_auc.append(sklearn.metrics.roc_auc_score(train_y,train_inv_y_score))

        test_inv_y_score = 1 - test_y_score
        val_inv_auc.append(sklearn.metrics.roc_auc_score(test_y,test_inv_y_score))

        plt.figure()

        fpr,tpr,_ = sklearn.metrics.roc_curve(train_y, train_y_score)
        plt.plot(fpr,tpr,label="train")
        fpr,tpr,_ = sklearn.metrics.roc_curve(test_y, test_y_score)
        plt.plot(fpr,tpr,label="val")
        plt.legend()
        plt.xlabel("False positive rate")
        plt.ylabel("True positive rate")

        plt.savefig(f"../vis/CROHN2/roc_curve_{i}.png")
        plt.close()
        '''
        
    #sys.exit(0)
    train_acc,train_auc = np.array(train_acc),np.array(train_auc)
    val_acc,val_auc = np.array(val_acc),np.array(val_auc)

    train_inv_auc = np.array(train_inv_auc)
    val_inv_auc = np.array(val_inv_auc)

    return {"train_acc":train_acc,"train_auc":train_auc,"val_acc":val_acc,"val_auc":val_auc,"train_inv_auc":train_inv_auc,"val_inv_auc":val_inv_auc}

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