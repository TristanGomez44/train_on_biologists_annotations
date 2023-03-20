from args import ArgReader
from args import str2bool
import os,sys
import glob
import torch
import torchvision
import numpy as np
import pandas as pd 
from matplotlib.colors import Normalize
import matplotlib.pyplot as plt
from matplotlib import cm 
plt.switch_backend('agg')
import umap.umap_ as umap
from PIL import ImageFont,Image,ImageDraw
from sklearn.manifold import TSNE
import sklearn
from sklearn import svm ,neural_network,tree,neighbors
from sklearn.manifold import  TSNE
from skimage.transform import resize
from scipy.stats import kendalltau
import load_data

def compNorm(featPath):

    if os.path.exists(featPath):
        features = np.load(featPath,mmap_mode="r+")
        nbFeat = features.shape[0]
        splitLen = [100*(i+1) for i in range(nbFeat//100)]
        features_split = np.split(features,splitLen)

        allNorm = None
        for feat in features_split:
            norm = np.sqrt(np.power(feat.astype(float),2).sum(axis=1,keepdims=True))
            if allNorm is None:
                allNorm = norm
            else:
                allNorm = np.concatenate((allNorm,norm),axis=0)
    else:
        allNorm = torch.ones((1,1,1,1))
    return allNorm

def find_saliency_maps(viz_id,model_ids,exp_id,expl):
    mapPaths = []
    suff = "" if viz_id == "" else "{}_".format(viz_id)
    for j in range(len(model_ids)):

        if expl[j] not in ["attMaps","norm"]:
            paths = glob.glob(f"../results/{exp_id}/saliencymaps_{expl[j]}_{model_ids[j]}_epoch*_{suff}.npy")
        else:
            print(f"../results/{exp_id}/{expl[j]}_{model_ids[j]}_epoch*.npy")
            paths = glob.glob(f"../results/{exp_id}/{expl[j]}_{model_ids[j]}_epoch*.npy")

        if len(paths) != 1:
            raise ValueError(f"Wrong paths number for {model_ids[j]}-{expl[j]}:",paths)

        mapPaths.append(paths[0])
    return mapPaths 

def select_images(args,class_index,inds,img_nb):
    _,testDataset = load_data.buildTestLoader(args,"test")
    maxInd = len(glob.glob(os.path.join("../data/",args.dataset_test,"*/*.*")))

    if len(inds) == 0:
        #Looking for the image at which the class we want begins
        if not class_index is None:
            startInd = 0

            classes = sorted(map(lambda x:x.split("/")[-2],glob.glob("../data/{}/*/".format(args.dataset_test))))
            for ind in range(class_index):
                className = classes[ind]
                startInd += len(glob.glob("../data/{}/{}/*".format(args.dataset_test,className)))

            className = classes[class_index]

            endInd = startInd + len(glob.glob("../data/{}/{}/*".format(args.dataset_test,className)))

        else:
            startInd = 0
            endInd = maxInd

        torch.manual_seed(1)
        inds = torch.randperm(endInd-startInd)[:img_nb]+startInd
   
    print("inds",inds)

    #In case there is not enough images
    img_nb = min(len(inds),img_nb)

    imgBatch = torch.cat([testDataset[ind][0].unsqueeze(0) for ind in inds],dim=0)
    return inds,imgBatch 

def load_feature_norm(model_ids,mapPaths,pond_by_norm,only_norm,expl):
    normDict = {}
    model_ids = np.array(model_ids)
    for j in range(len(mapPaths)):
        if (pond_by_norm[j] or only_norm[j]) and expl[j] == "attMaps":

            #Checking if the norm has already been loaded
            matchInds = np.argwhere(model_ids[:j]==model_ids[j])
            if len(matchInds) > 0 and normDict[matchInds[0,0]] is not None:
                normDict[j] = normDict[matchInds[0,0]]
            else:
                if not os.path.exists(mapPaths[j].replace("attMaps","norm")):
                    normDict[j] = compNorm(mapPaths[j].replace("attMaps","features"))
                    np.save(mapPaths[j].replace("attMaps","norm"),normDict[j])
                else:
                    normDict[j] = np.load(mapPaths[j].replace("attMaps","norm"))
                if len(normDict[j].shape) == 3:
                    normDict[j] = normDict[j][:,np.newaxis]
        else:
            normDict[j] = None
    return normDict

def showSalMaps(exp_id,img_nb,plot_id,nrows,class_index,inds,viz_id,args,
                    model_ids,expl,maps_inds,pond_by_norm,only_norm,interp,direct_ind,
                    sparsity_factor):

    gridImage = None
    args.normalize_data = False
    args.val_batch_size = img_nb
    fnt = ImageFont.truetype("arial.ttf", 40)
    cmPlasma = plt.get_cmap('plasma')
    imgSize = 448
    ptsImage = torch.zeros((3,imgSize,imgSize))

    mapPaths = find_saliency_maps(viz_id,model_ids,exp_id,expl)
    inds,imgBatch = select_images(args,class_index,inds,img_nb)
    img_nb = min(len(inds),img_nb)
    normDict = load_feature_norm(model_ids,mapPaths,pond_by_norm,only_norm,expl)

    for i in range(img_nb):

        if i % 10 == 0:
            print("i",i)

        img = imgBatch[i:i+1]
        img = (img-img.min())/(img.max()-img.min())

        if args.print_ind:
            imgPIL = Image.fromarray((255*img[0].permute(1,2,0).numpy()).astype("uint8"))
            imgDraw = ImageDraw.Draw(imgPIL)
            rectW = 180
            imgDraw.rectangle([(0,0), (rectW, 40)],fill="white")
            imgDraw.text((0,0), str(inds[i])+" ", font=fnt,fill=(0,0,0))
            img = torch.tensor(np.array(imgPIL)).permute(2,0,1).unsqueeze(0).float()/255

        if gridImage is None:
            gridImage = img
        else:
            gridImage = torch.cat((gridImage,img),dim=0)

        for j in range(len(mapPaths)):
            
            all_attMaps = np.load(mapPaths[j],mmap_mode="r")
            print(mapPaths[j],all_attMaps.shape,all_attMaps.min(),all_attMaps.max())
            attMap = all_attMaps[i] if direct_ind[j] else all_attMaps[inds[i]]

            if attMap.shape[0] != 1 and maps_inds[j] != -1:
                attMap = attMap[maps_inds[j]:maps_inds[j]+1]

            if (pond_by_norm[j] or only_norm[j]) and expl[j] == "attMaps":
                if direct_ind[j]:
                    norm = normDict[j][i]
                else:
                    norm = normDict[j][inds[i]]
                norm = (norm-norm.min())/(norm.max()-norm.min())

                if norm.shape[1:] != attMap.shape[1:]:
                    norm = resize(np.transpose(norm,(1,2,0)), (attMap.shape[1],attMap.shape[2]),anti_aliasing=True,mode="constant",order=0)
                    norm = np.transpose(norm,(2,0,1))

                attMap = norm*attMap if pond_by_norm[j] else norm
            
            attMap = (attMap-attMap.min())/(attMap.max()-attMap.min())

            if sparsity_factor[j] != 1:
                attMap = torch.pow(torch.from_numpy(attMap),sparsity_factor[j])
                attMap = (attMap-attMap.min())/(attMap.max()-attMap.min())

            if attMap.shape[0] == 1:
                attMap = cmPlasma(attMap[0])[:,:,:3]
            else:
                attMap = np.transpose(attMap,(1,2,0))

            ptsImageCopy = ptsImage.clone()
            interpOrder = 1 if interp[j] else 0
            ptsImageCopy = torch.tensor(resize(attMap, (ptsImageCopy.shape[1],ptsImageCopy.shape[2]),\
                                            anti_aliasing=True,mode="constant",order=interpOrder))
            ptsImageCopy = ptsImageCopy.permute(2,0,1).float().unsqueeze(0)
            
            img_gray = img.mean(dim=1,keepdim=True)
            img_gray = (img_gray-img_gray.min())/(img_gray.max()-img_gray.min())
            ptsImageCopy = 0.8*ptsImageCopy+0.2*img_gray
            gridImage = torch.cat((gridImage,ptsImageCopy),dim=0)

    outPath = "../vis/{}/{}.png".format(exp_id,plot_id)
    torchvision.utils.save_image(gridImage, outPath, nrow=(len(model_ids)+1)*nrows)
    os.system("convert  -resize 20% {} {}".format(outPath,outPath.replace(".png","_small.png")))

def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    ####################################### Show sal maps ##################################
    argreader.parser.add_argument('--show_maps',action="store_true") 
    
    argreader.parser.add_argument('--img_nb',type=int,default=100)
    argreader.parser.add_argument('--plot_id',type=str,metavar="ID",help='The plot id',default="")
    argreader.parser.add_argument('--nrows',type=int,metavar="INT",help='The number of rows',default=4)
    argreader.parser.add_argument('--class_index',type=int,metavar="INT",help='The class index to show')
    argreader.parser.add_argument('--viz_id',type=str,help='The viz ID to plot gradcam like viz',default="")
    argreader.parser.add_argument('--print_ind',type=str2bool,metavar="BOOL",help='To print image index',default=False)

    argreader.parser.add_argument('--model_ids',type=str,metavar="IDS",nargs="*",help='The list of model ids.')
    argreader.parser.add_argument('--maps_inds',type=int,nargs="*",metavar="INT",help='The index of the attention map to use\
                                     when there is several. If there only one or if there is none, set this to -1',default=[])
    argreader.parser.add_argument('--inds',type=int,nargs="*",metavar="INT",help='The index of the images to keep',default=[])
    argreader.parser.add_argument('--interp',type=str2bool,nargs="*",metavar="BOOL",help='To smoothly interpolate the att map.',default=[])
    argreader.parser.add_argument('--direct_ind',type=str2bool,nargs="*",metavar="BOOL",help='To use direct indices',default=[])
    argreader.parser.add_argument('--pond_by_norm',type=str2bool,nargs="*",metavar="BOOL",help='To also show the norm of pixels along with the attention weights.',default=[])
    argreader.parser.add_argument('--only_norm',type=str2bool,nargs="*",metavar="BOOL",help='To only plot the norm of pixels',default=[])
    argreader.parser.add_argument('--expl',type=str,nargs="*",metavar="BOOL",help='The explanation type',default=[])
    argreader.parser.add_argument('--sparsity_factor',type=float,nargs="*",metavar="BOOL",help='Set this arg to modify the sparsity of attention maps',default=[])

    argreader = load_data.addArgs(argreader)

    #Reading the comand line arg
    argreader.getRemainingArgs()

    #Getting the args from command line and config file
    args = argreader.args

    #Setting default values
    default_values = {"pond_by_norm":True,"only_norm":False,"interp":False,"direct_ind":False,"maps_inds":-1,
                        "sparsity_factor":1}
    for key in default_values:
        param = getattr(args,key)
        if len(param) == 0:
            value = default_values[key]
        elif len(param) == 1:
            value = param[0]

        param = [value for _ in range(len(args.model_ids))]

        setattr(args,key,param)

    showSalMaps(args.exp_id,args.img_nb,args.plot_id,args.nrows,args.class_index,args.inds,args.viz_id,args,
                args.model_ids,args.expl,args.maps_inds,args.pond_by_norm,args.only_norm,args.interp,args.direct_ind,
                args.sparsity_factor)

if __name__ == "__main__":
    main()
