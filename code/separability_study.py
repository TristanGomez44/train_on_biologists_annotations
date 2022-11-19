
import glob,os 

import numpy as np 
import torch
from sklearn import svm
import sklearn.metrics

from trainVal import addInitArgs,addValArgs,getBatch,preprocessAndLoadParams
import modelBuilder,load_data
import args
from args import ArgReader
from args import str2bool

def main(argv=None):
    # Getting arguments from config file and command line
    # Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--transf1', type=str,default="identity")
    argreader.parser.add_argument('--transf2', type=str,default="identity")
    argreader.parser.add_argument('--model', type=str,default="SVM")

    argreader = addInitArgs(argreader)
    argreader = addValArgs(argreader)

    argreader = modelBuilder.addArgs(argreader)
    argreader = load_data.addArgs(argreader)

    # Reading the comand line arg
    argreader.getRemainingArgs()

    args = argreader.args

    repres1 = np.load(f"../results/{args.exp_id}/img_repr_model{args.model_id}_transf{args.transf1}.npy")
    repres2 = np.load(f"../results/{args.exp_id}/img_repr_model{args.model_id}_transf{args.transf2}.npy")

    labels1 = np.zeros(len(repres1)).astype("int")
    labels2 = np.ones(len(repres2)).astype("int")

    torch.manual_seed(args.seed)
    inds = np.random.permutation(len(repres1))
    repres1,labels1 = repres1[inds],labels1[inds]
    repres2,labels2 = repres2[inds],labels2[inds]

    train_x = np.concatenate((repres1[:len(repres1)//2],repres2[:len(repres2)//2]),axis=0)
    test_x = np.concatenate((repres1[len(repres1)//2:],repres2[len(repres2)//2:]),axis=0)

    train_y = np.concatenate((labels1[:len(repres1)//2],labels2[:len(repres2)//2]),axis=0)
    test_y = np.concatenate((labels1[len(repres1)//2:],labels2[len(repres2)//2:]),axis=0)

    constDic = {"SVM":svm.SVC}
    kwargsDic = {"SVM":{"probability":True}}

    model = constDic[args.model](**kwargsDic[args.model])

    model.fit(train_x,train_y)

    train_y_score = model.predict_proba(train_x)[:,1]
    train_acc = model.score(train_x,train_y)
    train_auc = sklearn.metrics.roc_auc_score(train_y,train_y_score)
    
    test_y_score = model.predict_proba(test_x)[:,1]
    val_acc = model.score(test_x,test_y)            
    val_auc = sklearn.metrics.roc_auc_score(test_y,test_y_score)

    csv_path = f"../results/{args.exp_id}/separability_study.csv"
    if not os.path.exists(csv_path):
        with open(csv_path,"w") as file:
            print(f"model_id,tranfs1,transf2,sec_model,train_acc,train_auc,val_acc,val_auc",file=file) 

    with open(csv_path,"a") as file:
        print(f"{args.model_id},{args.transf1},{args.transf2},{args.model},{train_acc},{train_auc},{val_acc},{val_auc}",file=file)

if __name__ == "__main__":
    main()