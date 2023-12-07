
import os

import numpy as np
import torch
import cv2
import torch.nn.functional as F

from args import ArgReader,addValArgs,init_post_hoc_arg,addSalMetrArgs
import modelBuilder
import load_data
import utils 
import trainVal
from utils import get_feature_and_output,get_datasets_and_model
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def cos_sim(a,b):
    #a = a[:,None,:]
    #b = b[None,:,:]
    a = torch.tensor(a)
    b = torch.tensor(b)
    print("cosim",a.shape,b.shape)
    C = F.normalize(a) @ F.normalize(b).t()
    print("cosim",C.shape)
    return C
    #print("cosim",a.shape,b.shape)
    #return torch.nn.functional.cosine_similarity(a,b,axis=axis).numpy()

def find_closest_neighbor(result_dic,args):

    closest_train_idx_path = f"../results/{args.exp_id}/closest_train_idx_{args.model_id}.npy"

    print(closest_train_idx_path)

    if not os.path.exists(closest_train_idx_path):

        labels= np.unique(result_dic["train"]["target"])
        closest_train_idx = []

        for label in labels:
            test_idx = np.where(result_dic["test"]["target"] == label)[0]
            test_pooled_features = result_dic["test"]["pooled_features"][test_idx]

            train_idx = np.where(result_dic["train"]["target"] == label)[0]
            train_pooled_features = result_dic["train"]["pooled_features"][train_idx]

            print(result_dic["test"]["pooled_features"][0].shape,test_idx,result_dic["train"]["pooled_features"][0].shape,train_idx)
            dist = -cos_sim(test_pooled_features,train_pooled_features)

            closest_train_idx.extend(train_idx[np.argmin(dist,axis=-1)])

        closest_train_idx = np.array(closest_train_idx)
        np.save(closest_train_idx_path,closest_train_idx)
            
    else:
        closest_train_idx = np.load(closest_train_idx_path,allow_pickle=True)

    return closest_train_idx

def similarity_map(train_feat,test_feat,maps_shape,sim_map="mean_activation"):

    train_feat = train_feat.transpose()
    test_feat = test_feat.transpose()

    train_feat = train_feat[:,None,:]
    test_feat = test_feat[None,:,:]

    if sim_map == "dot":
        sim_matrix = (train_feat*test_feat).sum(axis=-1)
        train_sim_map = sim_matrix.max(axis=-1)
        test_sim_map = sim_matrix.max(axis=0)
    elif sim_map == "mean_activation":
        train_sim_map = train_feat.mean(axis=-1)
        print("mean_act",train_feat.shape,train_sim_map.shape)
        test_sim_map = test_feat.mean(axis=-1)
    else:
        raise ValueError("Unkown sim map method",sim_map)

    print(train_sim_map.shape)
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
            
            print("similarity_maps() test_feat",test_feat.shape,train_feat.shape,closest_train_idx[i])

            #Stores maps shape 
            maps_shape = test_feat.shape[1:]
            #print("ORIGINAL SHAPE",test_feat.shape)
            old_shape = train_feat.shape
            train_feat = train_feat.reshape(train_feat.shape[0],-1)
            
            test_feat = test_feat.reshape(test_feat.shape[0],-1)

            print("train_feat",train_feat.shape,"test_feat",test_feat.shape)
            train_map,test_map = similarity_map(train_feat,test_feat,maps_shape)

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
    sal_map = cv2.resize(sal_map, (img.shape[2], img.shape[1]),interpolation = cv2.INTER_CUBIC).astype(np.float32)
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

        test_idx = i
        train_idx = closest_train_idx[i]

        test_img = test_dataset[test_idx][0]
        train_img = train_dataset[train_idx][0]

        test_img = utils.inv_imgnet_norm(test_img)
        train_img = utils.inv_imgnet_norm(train_img)

        train_map = preproc_map(train_img,train_maps[i],cmap)
        test_map = preproc_map(test_img,test_maps[i],cmap)

        grid.append(test_img)
        grid.append(test_map)
        grid.append(train_img)
        grid.append(train_map)
    
        if args.debug and i==5:
            break
        
        if i % 5 == 0:
            print(f"{i}/{len(closest_train_idx)}")

            grid = torch.from_numpy(np.stack(grid,0))
            fig_path = f"../vis/{args.exp_id}/similarity_map_{args.model_id}_{i}.png"
            utils.save_image(grid,fig_path,row_nb=len(grid)//4,apply_inv_norm=False)

            grid = []

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

    argreader.parser.add_argument('--task_to_train', type=str, default="all")
    argreader.parser.add_argument('--task_output_to_save', type=str, default="ICM")

    # Reading the comand line arg
    argreader.getRemainingArgs()

    args = argreader.args

    train_loader,train_dataset,test_loader,test_dataset,net = get_datasets_and_model(args)

    result_dic = get_feature_and_output(train_loader,test_loader,net,args)

    closest_train_idx = find_closest_neighbor(result_dic,args)

    train_maps,test_maps = similarity_maps(args,result_dic,closest_train_idx)

    show_similarity_map(args,test_dataset,train_dataset,closest_train_idx,train_maps,test_maps)

if __name__ == "__main__":
    main()