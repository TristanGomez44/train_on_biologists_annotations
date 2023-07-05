
import os
import sys
import glob

from shutil import copyfile
import gc
import subprocess

import numpy as np
import torch
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import optuna
import sqlite3
import cv2

from utils import _remove_no_annot
from args import ArgReader,str2bool,addInitArgs,addValArgs,init_post_hoc_arg,addLossTermArgs,addSalMetrArgs
import init_model
from loss import SupervisedLoss,SelfSuperVisedLoss,agregate_losses
import modelBuilder
import load_data
import metrics
import update
import utils 
import trainVal

def get_feature_and_output(train_loader,test_loader,net,args):
    result_dic = {}
    for label,loader in zip(["train","test"],[train_loader,test_loader]):
        pooled_features_path = f"../results/{args.exp_id}/pooled_features_{args.model_id}_{label}.npy"
        
        result_dic[label] = {}
        if not os.path.exists(pooled_features_path):
            print(f"Inference on {label} set")
            pooled_features = []
            features = []
            output = []
            target = []
            for idx,batch in enumerate(loader):
        
                if idx%10 == 0:
                    print(f"\tProcessing batch {idx}/{len(loader)}")

                image,annot_dict = batch
                output_dic = net(image)        
            
                pooled_features.append(output_dic["feat_pooled"].cpu().numpy())
                features.append(output_dic["feat"].cpu().numpy())
                output.append(output_dic["output_"+args.task].cpu().numpy())
                target.append(annot_dict[args.task].cpu().numpy())

                if args.debug:
                    break

            pooled_features = np.concatenate(pooled_features,axis=0)
            features = np.concatenate(features,axis=0)
            output = np.concatenate(output,axis=0)
            target = np.concatenate(target,axis=0)

            np.save(pooled_features_path,pooled_features)
            np.save(pooled_features_path.replace("pooled_features","features"),features)
            np.save(pooled_features_path.replace("pooled_features","output"),output)
            np.save(pooled_features_path.replace("pooled_features","target"),target)

        else:
            pooled_features = np.load(pooled_features_path)
            features = np.load(pooled_features_path.replace("pooled_features","features"))
            output = np.load(pooled_features_path.replace("pooled_features","output"))
            target = np.load(pooled_features_path.replace("pooled_features","target"))

        result_dic[label]["pooled_features"] = pooled_features
        result_dic[label]["features"] = features
        result_dic[label]["output"] = output
        result_dic[label]["target"] = target
    
    return result_dic

def find_closest_neighbor(result_dic,args):

    closest_train_idx_path = f"../results/{args.exp_id}/closest_train_idx_{args.model_id}.npy"

    if not os.path.exists(closest_train_idx_path):

        labels= np.unique(result_dic["train"]["target"])
        closest_train_idx = []

        for label in labels:
            test_idx = np.where(result_dic["test"]["target"] == label)[0]
            test_pooled_features = result_dic["test"]["pooled_features"][test_idx]

            train_idx = np.where(result_dic["train"]["target"] == label)[0]
            train_pooled_features = result_dic["train"]["pooled_features"][train_idx]

            dist = np.linalg.norm(test_pooled_features[:,None,:] - train_pooled_features[None,:,:],axis=-1)

            closest_train_idx.extend(train_idx[np.argmin(dist,axis=-1)])

        closest_train_idx = np.array(closest_train_idx)
        np.save(closest_train_idx_path,closest_train_idx)
            
    else:
        closest_train_idx = np.load(closest_train_idx_path)

    return closest_train_idx

def similarity_map(feat_ref,feat,orig_map_shape):
    most_active_feat = feat_ref[:,np.argmax(feat_ref.sum(axis=0))][:,None]

    most_active_feat = most_active_feat.transpose()
    feat = feat.transpose()

    #cosine similarity between the most active feature and the other features
    sim_map = (most_active_feat*feat).sum(axis=1)/(np.linalg.norm(most_active_feat,axis=1)*np.linalg.norm(feat,axis=1))

    sim_map = sim_map.reshape((orig_map_shape))
    return sim_map
        
def similarity_maps(args,result_dic,closest_train_idx):
    #For each train example - test example pair,
    #compute the similarity map between the most active train feature and the test features 
    #and the similarity map between the most active test feature and the train features
    train_to_test_path = f"../results/{args.exp_id}/train_to_test_{args.model_id}.npy"
    test_to_train_path = f"../results/{args.exp_id}/test_to_train_{args.model_id}.npy"
    
    if not os.path.exists(train_to_test_path) or not os.path.exists(test_to_train_path):
        
        train_to_test = []
        test_to_train = []
        for i in range(len(closest_train_idx)):

            test_feat = result_dic["test"]["features"][i]
            train_feat = result_dic["train"]["features"][closest_train_idx[i]]
            
            #Stores maps shape 
            maps_shape = test_feat.shape[1:]

            test_feat = test_feat.reshape(test_feat.shape[0],-1)
            train_feat = train_feat.reshape(train_feat.shape[0],-1)

            train_to_test.append(similarity_map(train_feat,test_feat,maps_shape))
            test_to_train.append(similarity_map(test_feat,train_feat,maps_shape))
            
        train_to_test = np.array(train_to_test)
        test_to_train = np.array(test_to_train)

        np.save(train_to_test_path,train_to_test)
        np.save(test_to_train_path,test_to_train)

    else:
        train_to_test = np.load(train_to_test_path)
        test_to_train = np.load(test_to_train_path)

    return train_to_test,test_to_train

def preproc_map(img,sal_map,cmap):
    sal_map = utils.normalize_tensor(sal_map)
    sal_map = cmap(sal_map)[:,:,:3]
    sal_map = cv2.resize(sal_map, (img.shape[2], img.shape[1])).astype(np.float32)
    sal_map = sal_map.transpose(2,0,1)
    img = img.mean(axis=0,keepdim=True).expand(3,-1,-1)  
    sal_map = 0.2*img + 0.8*sal_map
    return sal_map

def show_similarity_map(args,test_dataset,train_dataset,closest_train_idx,train_to_test,test_to_train,cmap="plasma"):
    #For each train example - test example pair,
    #Show the train image, the test image, the similarity map between the most active train feature and the test features 
    #and the similarity map between the most active test feature and the train features
    grid = []

    cmap = plt.get_cmap(cmap)

    for i in range(len(closest_train_idx)):

        if i % 20 == 0:
            print(f"{i}/{len(closest_train_idx)}")

        test_idx = i
        train_idx = closest_train_idx[i]

        test_img = test_dataset[test_idx][0]
        train_img = train_dataset[train_idx][0]

        test_img = utils.inv_imgnet_norm(test_img)
        train_img = utils.inv_imgnet_norm(train_img)

        test_to_train_map = preproc_map(train_img,test_to_train[i],cmap)
        train_to_test_map = preproc_map(test_img,train_to_test[i],cmap)

        grid.append(test_img)
        grid.append(test_to_train_map)
        grid.append(train_img)
        grid.append(train_to_test_map)
    
        if args.debug and i==5:
            break

    grid = torch.from_numpy(np.stack(grid,0))
    fig_path = f"../vis/{args.exp_id}/similarity_map_{args.model_id}.png"
    utils.save_image(grid,fig_path,row_nb=len(grid)//4,apply_inv_norm=False)

def main(argv=None):
    # Getting arguments from config file and command line
    # Building the arg reader
    argreader = ArgReader(argv)

    argreader = trainVal.addInitArgs(argreader)
    argreader = trainVal.addSSLArgs(argreader)
    argreader = trainVal.addRegressionArgs(argreader)
    argreader = modelBuilder.addArgs(argreader)
    argreader = load_data.addArgs(argreader)

    argreader = addValArgs(argreader)
    argreader = addSalMetrArgs(argreader)
    argreader = init_post_hoc_arg(argreader)

    argreader.parser.add_argument('--task', type=str, default="icm")

    # Reading the comand line arg
    argreader.getRemainingArgs()

    args = argreader.args

    train_loader,train_dataset = load_data.buildTestLoader(args,"train")
    test_loader,test_dataset = load_data.buildTestLoader(args,"test")

    net = modelBuilder.netBuilder(args)
    best_paths = glob.glob(f"../models/{args.exp_id}/model{args.model_id}_best_epoch*")
    assert len(best_paths) == 1, "There should be only one best model"
    best_path = best_paths[0]
    net = init_model.preprocessAndLoadParams(best_path,args.cuda,net,ssl=args.ssl)
    net.eval()
    torch.set_grad_enabled(False)

    result_dic = get_feature_and_output(train_loader,test_loader,net,args)

    closest_train_idx = find_closest_neighbor(result_dic,args)

    train_to_test,test_to_train = similarity_maps(args,result_dic,closest_train_idx)

    show_similarity_map(args,test_dataset,train_dataset,closest_train_idx,train_to_test,test_to_train)

if __name__ == "__main__":
    main()