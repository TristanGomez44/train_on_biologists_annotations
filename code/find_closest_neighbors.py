
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

def similarity_map(train_feat,test_feat,maps_shape):

    train_feat = train_feat.transpose()
    test_feat = test_feat.transpose()

    train_feat = train_feat[:,None,:]
    test_feat = test_feat[None,:,:]

    #cosine similarity between the most active feature and the other features
    #sim_matrix = (train_feat*test_feat).sum(axis=-1)/(np.linalg.norm(train_feat,axis=-1)*np.linalg.norm(test_feat,axis=-1))
    sim_matrix = (train_feat*test_feat).sum(axis=-1)

    train_sim_map = sim_matrix.max(axis=-1)
    test_sim_map = sim_matrix.max(axis=0)

    train_sim_map = train_sim_map.reshape(maps_shape)
    test_sim_map = test_sim_map.reshape(maps_shape)

    return train_sim_map,test_sim_map
        
def similarity_maps(args,result_dic,closest_train_idx):
    #For each train example - test example pair,
    #compute the similarity map between the most active train feature and the test features 
    #and the similarity map between the most active test feature and the train features
    test_maps_path = f"../results/{args.exp_id}/test_maps_{args.model_id}.npy"
    train_maps_path = f"../results/{args.exp_id}/train_maps_{args.model_id}.npy"
    
    if not os.path.exists(test_maps_path) or not os.path.exists(train_maps_path):
        
        test_maps = []
        train_maps = []
        for i in range(len(closest_train_idx)):

            test_feat = result_dic["test"]["features"][i]
            train_feat = result_dic["train"]["features"][closest_train_idx[i]]
            
            #Stores maps shape 
            maps_shape = test_feat.shape[1:]

            test_feat = test_feat.reshape(test_feat.shape[0],-1)
            train_feat = train_feat.reshape(train_feat.shape[0],-1)

            train_map,test_map= similarity_map(train_feat,test_feat,maps_shape)

            train_maps.append(train_map)
            test_maps.append(test_map)

        test_maps = np.array(test_maps)
        train_maps = np.array(train_maps)

        np.save(test_maps_path,test_maps)
        np.save(train_maps_path,train_maps)

    else:
        test_maps = np.load(test_maps_path)
        train_maps = np.load(train_maps_path)

    return train_maps,test_maps

def preproc_map(img,sal_map,cmap):
    sal_map = utils.normalize_tensor(sal_map)
    sal_map = cmap(sal_map)[:,:,:3]
    sal_map = cv2.resize(sal_map, (img.shape[2], img.shape[1]),interpolation = cv2.INTER_NEAREST).astype(np.float32)
    sal_map = sal_map.transpose(2,0,1)
    img = img.mean(axis=0,keepdim=True).expand(3,-1,-1)  
    sal_map = 0.2*img + 0.8*sal_map
    return sal_map

def show_similarity_map(args,test_dataset,train_dataset,closest_train_idx,train_maps,test_maps,cmap="plasma"):
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

        train_map = preproc_map(train_img,train_maps[i],cmap)
        test_map = preproc_map(test_img,test_maps[i],cmap)

        grid.append(test_img)
        grid.append(train_map)
        grid.append(train_img)
        grid.append(test_map)
    
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

    train_maps,test_maps = similarity_maps(args,result_dic,closest_train_idx)

    show_similarity_map(args,test_dataset,train_dataset,closest_train_idx,train_maps,test_maps)

if __name__ == "__main__":
    main()