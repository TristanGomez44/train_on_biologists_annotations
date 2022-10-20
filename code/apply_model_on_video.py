import glob,sys,os
import numpy as np

import torch,torchvision
from torchvision import transforms
from torchvision.io import read_video
from PIL import Image 

import args
from args import ArgReader
from args import str2bool
from trainVal import addInitArgs,addOptimArgs,addValArgs,addLossTermArgs,preprocessAndLoadParams
from modelBuilder import addArgs as addArgsBuilder,netBuilder
from load_data import addArgs as addArgsLoad,get_img_size

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
        
            if args.debug:
                break

    attMaps = torch.cat(attMaps,dim=0).cpu().numpy()
    preds = torch.cat(preds,dim=0).cpu().numpy()

    vidName = get_video_name(args.img_folder)
    np.save(f"../results/{args.exp_id}/{vidName}_{args.model_id}_attMaps.npy",attMaps)
    np.save(f"../results/{args.exp_id}/{vidName}_{args.model_id}_preds.npy",preds)
   
if __name__ == "__main__":
    main()








        
        


