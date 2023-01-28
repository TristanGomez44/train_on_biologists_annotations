import glob,sys,os
import numpy as np

from torch.nn.functional import cosine_similarity
import torch,torchvision
from torchvision import transforms
from torchvision.io import read_video
from PIL import Image 
from scipy.optimize import linear_sum_assignment
import matplotlib.pyplot as plt 

import args
from args import ArgReader
from args import str2bool
from trainVal import addInitArgs,addOptimArgs,addValArgs,addLossTermArgs,preprocessAndLoadParams
from modelBuilder import addArgs as addArgsBuilder,netBuilder
from load_data import addArgs as addArgsLoad,get_img_size

def apply_perm_and_norm_sort(feats,attMaps,assign_list,break_inds):
    current_group_ind = len(break_inds) - 1
    for i in range(len(feats)-1,0,-1):

        if i == break_inds[current_group_ind] - 1:
            current_group_ind -= 1

        if current_group_ind == len(break_inds) - 1:
            start = break_inds[len(break_inds)- 1]
            feats_group = feats[start:]
            attMaps_group = attMaps[start:]
        else:
            start,end = break_inds[current_group_ind],break_inds[current_group_ind+1]
            feats_group = feats[start:end]
            attMaps_group = attMaps[start:end]
        
        feats_group = feats_group[i - start:,assign_list[i-1][1]]
        attMaps_group = attMaps_group[i - start:,assign_list[i-1][1]]

        #Sorting the features to put the vectors with the highest mean norm in the first slot,
        # the second highest in the second slot, etc.
        if i == break_inds[current_group_ind]:
            sorted_slot_ind = torch.abs(feats_group).sum(dim=2).mean(dim=0).argsort(descending=True)
            feats_group = feats_group[:,sorted_slot_ind]
            attMaps_group = attMaps_group[:,sorted_slot_ind]

        feats[i:i+len(feats_group)] = feats_group
        attMaps[i:i+len(feats_group)] = attMaps_group
    return feats,attMaps

def compute_perm_and_groups(feats,attMaps,thres,att_maps_sim):

    if att_maps_sim:
        attMaps_flat = attMaps.view(attMaps.shape[0],attMaps.shape[1],-1)
        dist = torch.cdist(attMaps_flat[:-1],attMaps_flat[1:])
        costs = dist.cpu().numpy()
    else:
        cos_sim = cosine_similarity(feats[:-1].unsqueeze(1),feats[1:].unsqueeze(2),dim=3)
        costs = -cos_sim.cpu().numpy()

    assign_list = []
    break_inds = [0]
    opt_cost_list = []
    for i in range(len(feats)-1):
        assign = linear_sum_assignment(costs[i])
        assign_list.append(assign)
        opt_cost = costs[i][assign[0], assign[1]].mean()
        if opt_cost > thres:
            break_inds.append(i+1)

        opt_cost_list.append(opt_cost)

    assign_list = torch.from_numpy(np.array(assign_list))
    print(len(break_inds))
    return assign_list,break_inds

def load_batch(i,bs,transf,path_list):
    path_list_batch = path_list[i*bs:(i+1)*bs]
    batch_preprocessed = []
    for j in range(len(path_list_batch)):
        image = Image.open(path_list_batch[j]).convert('RGB') 
        img_preproc = transf(image)
        batch_preprocessed.append(img_preproc)
    batch_preprocessed = torch.stack(batch_preprocessed,dim=0)
    return batch_preprocessed

def get_transform(args):
    crop_ratio=args.crop_ratio
    resize = get_img_size(args)
    kwargs={"size":(int(resize / crop_ratio), int(resize / crop_ratio))}
    transf = [transforms.Resize(**kwargs),transforms.CenterCrop(resize)]
    transf.extend([transforms.ToTensor()])
    transf = transforms.Compose(transf)
    return transf 


def normalize(tens):
    return (tens - tens.min())/(tens.max() - tens.min())

def get_attMaps_or_featNorm(out_dic):

    norm = torch.sqrt(torch.pow(out_dic["features"],2).sum(dim=1,keepdim=True))
    norm = normalize(norm)

    if "attMaps" in out_dic:
        attMaps = out_dic["attMaps"]
        attMaps = normalize(attMaps)
        tens = normalize(attMaps*norm)
    else:
        tens = norm

    return tens

def get_video_name(path):
    path_split = path.split("/")
    if path_split[-1] == "":
        vidName = path_split[-2]
    else:
        vidName = path_split[-1]
    return vidName
    
def get_path_list(img_folder):
    find_img_ind = lambda x:int(x.split("RUN")[1].split(".")[0])
    path_list = sorted(glob.glob(os.path.join(img_folder,"F0","*.*")),key=find_img_ind)
    return path_list

def main(argv=None):
    # Getting arguments from config file and command line
    # Building the arg reader
    argreader = ArgReader(argv)
    
    argreader.parser.add_argument('--img_folder', type=str)
    argreader.parser.add_argument('--hungarian_alg',action="store_true")
    argreader.parser.add_argument('--att_maps_sim',action="store_true")
    argreader.parser.add_argument('--cosine_sim_thres',type=float,default=0.975)
    argreader.parser.add_argument('--eucl_dist_thres',type=float,default=2)

    argreader = addInitArgs(argreader)
    argreader = addOptimArgs(argreader)
    argreader = addValArgs(argreader)
    argreader = addLossTermArgs(argreader)
    argreader = addArgsBuilder(argreader)
    argreader = addArgsLoad(argreader)

    # Reading the comand line arg
    argreader.getRemainingArgs()

    args = argreader.args
    best_path = glob.glob("../models/{}/model{}_best_epoch*".format(args.exp_id, args.model_id))[0]
    net = netBuilder(args,gpu=0).eval()
    net = preprocessAndLoadParams(best_path,args.cuda and torch.cuda.is_available(),net)

    path_list = get_path_list(args.img_folder)
    video_len = len(path_list)
 
    bs = args.val_batch_size
    batch_nb = video_len//bs + (video_len % bs > 0)

    transf = get_transform(args)

    attMaps = []
    feats = []
    preds = []
    with torch.no_grad():
        for i in range(batch_nb):

            if i % 10 == 0:
                print(i+1,"/",batch_nb)

            batch_preprocessed = load_batch(i,bs,transf,path_list)

            out_dic = net(batch_preprocessed)

            attMaps.append(get_attMaps_or_featNorm(out_dic))
            pred = out_dic["pred"]
            preds.append(pred)

            feat = out_dic["x"]
            feats.append(feat)
        
            if args.debug:
                break

    attMaps = torch.cat(attMaps,dim=0)
    feats = torch.cat(feats,dim=0)
    preds = torch.cat(preds,dim=0)

    if args.hungarian_alg:
        thres = args.eucl_dist_thres if args.att_maps_sim else args.cosine_sim_thres
        att_map_nb = attMaps.shape[1]
        feats = feats.view(feats.shape[0],att_map_nb,feats.shape[1]//att_map_nb)
        assign_list,break_inds = compute_perm_and_groups(feats,attMaps,thres,args.att_maps_sim)
        feats,attMaps = apply_perm_and_norm_sort(feats,attMaps,assign_list,break_inds)
        feats = feats.view(feats.shape[0],-1)
        with torch.no_grad():
            preds = net.secondModel({"x":feats})["pred"]

    feats,attMaps,preds = feats.cpu().numpy(),attMaps.cpu().numpy(),preds.cpu().numpy()

    model_id_suff = ""
    if args.hungarian_alg:
        model_id_suff += "_hung"+str(thres)

        if args.att_maps_sim:
            model_id_suff += "attMapSim"

    vidName = get_video_name(args.img_folder)
    np.save(f"../results/{args.exp_id}/{vidName}_{args.model_id}{model_id_suff}_attMaps.npy",attMaps)
    np.save(f"../results/{args.exp_id}/{vidName}_{args.model_id}{model_id_suff}_preds.npy",preds)
   
if __name__ == "__main__":
    main()








        
        


