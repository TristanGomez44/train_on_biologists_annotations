import glob,sys,os
import numpy as np

import torch,torchvision
from torchvision import transforms
from torchvision.io import read_video
from PIL import Image 
import matplotlib.pyplot as plt 
from skimage.transform import resize 

import args
from args import ArgReader
from args import str2bool
from load_data import get_img_size
from apply_model_on_video import get_video_name,get_path_list

def normalize(arr):
    arr_min = arr.min(axis=(1,2,3),keepdims=True)
    arr_max = arr.max(axis=(1,2,3),keepdims=True)
    return (arr - arr_min)/(arr_max - arr_min)

def read_fold(path_list):
    img_list = []
    for i in range(len(path_list)):
        img_list.append(np.array(Image.open(path_list[i]).convert('RGB')))
    img_list = np.stack(img_list)    
    return img_list

def preprocess_attMaps(attMaps,all_maps):

    attMaps = normalize(attMaps)
    if all_maps and attMaps.shape[1] != 1:
        attMaps = 255*attMaps.transpose((0,2,3,1))
    else:    
        if attMaps.shape[1] != 1:
            attMaps = attMaps[:,0:1]
        cmPlasma = plt.get_cmap('plasma')
        shape = attMaps.shape 
        attMaps = 255*cmPlasma(attMaps.reshape(-1))[:,:3].reshape((shape[0],shape[2],shape[3],3))
    return attMaps 

def main(argv=None):
    argreader = ArgReader(argv)
    argreader.parser.add_argument('--img_folder', type=str)
    argreader.parser.add_argument('--all_maps', action="store_true")
    argreader.getRemainingArgs()
    args = argreader.args

    path_list = get_path_list(args.img_folder)
    vidName = get_video_name(args.img_folder)
    attMaps = np.load(f"../results/{args.exp_id}/{vidName}_{args.model_id}_attMaps.npy",mmap_mode="r")

    attMaps = preprocess_attMaps(attMaps,args.all_maps)

    bs = args.val_batch_size

    img_list = read_fold(path_list)
    img_size = get_img_size(args)

    all_img = []
    for i in range(len(attMaps)):

        img = resize(img_list[i], (img_size,img_size),order=0)
        attMap = resize(attMaps[i], (img_size,img_size),order=0)
        
        attMap = 0.8*attMap + 0.2*img

        #print(img.min().item(),img.mean().item(),img.max().item(),attMap.min().item(),attMap.mean().item(),attMap.max().item())
    
        img = np.concatenate((img,attMap),axis=1)
        all_img.append(img)
    all_img = torch.from_numpy(np.stack(all_img))

    fps = 5 if len(all_img) <= 40 else 20
    torchvision.io.write_video(f"../vis/{args.exp_id}/{vidName}_{args.model_id}_allMaps={args.all_maps}_attMaps.mp4",all_img,fps=fps)
    
if __name__ == "__main__":
    main()








        
        


