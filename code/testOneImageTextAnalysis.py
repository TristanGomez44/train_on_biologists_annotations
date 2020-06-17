import torch
import args
import modelBuilder
import load_data
import trainVal
from args import ArgReader
from tensorboardX import SummaryWriter
import torchvision
import sys
import transfSimCLR
import nt_xent
from args import str2bool
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import imageio
from skimage import img_as_ubyte
import matplotlib.cm as cm
import time
import skimage.feature
import torch_cluster
from skimage import img_as_ubyte
from scipy import ndimage
from skimage.transform import resize
import scipy.ndimage.filters as filters
import torch_geometric
import torch.nn.functional as F
from sklearn.manifold import TSNE
import umap
def main(argv=None):
    # Getting arguments from config file and command line
    # Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--layer', type=int,
                                  help="The encoding layer to use for feature similarity.",default=4)

    argreader.parser.add_argument('--last_conv_decod', type=str2bool,
                                  help="Compute the prob map on the last conv layer of the decoder",default=False)

    argreader.parser.add_argument('--lay_1_conv_decod', type=str2bool,
                                  help="Compute the prob map on the feature of layer 1 of the decoder",default=False)

    argreader.parser.add_argument('--patch_sim', type=str2bool,
                                  help="To run a small experiment with an untrained CNN",default=False)
    argreader.parser.add_argument('--patch_sim_resnet', type=str2bool,
                                  help="To use resnet for the patch similarity experiment",default=False)
    argreader.parser.add_argument('--patch_sim_usemodel', type=str2bool,
                                  help="Set this to False to compare directly pixels",default=True)

    argreader.parser.add_argument('--patch_sim_restype', type=str,
                                  help="The resnet to use",default="resnet18")

    argreader.parser.add_argument('--patch_sim_relu_on_simple', type=str2bool,
                                  help="To add relu when using the simple CNN",default=False)

    argreader.parser.add_argument('--patch_sim_pretr_res', type=str2bool,
                                  help="To have the patch sim resnet pretrained on Imagenet",default=False)
    argreader.parser.add_argument('--patch_size', type=int,
                                  help="The patch size for the patch similarity exp",default=50)
    argreader.parser.add_argument('--patch_stride', type=int,
                                  help="The patch stride for the patch similarity exp",default=50)

    argreader.parser.add_argument('--patch_sim_full', type=str2bool,
                                  help="To compute similarity with every patch in the image, not just the neighbors.",default=True)
    argreader.parser.add_argument('--patch_sim_neig_mode_ker_size', type=int,
                                  help="The kernel size when working in neighbor mode",default=3)

    argreader.parser.add_argument('--patch_sim_paral_mode', type=str2bool,
                                  help="To compute patch similarity in parralel (requries more ram but faster) instead of in sequential.",default=True)

    argreader.parser.add_argument('--patch_sim_group_nb', type=int,
                                  help="To compute gram matrix by grouping features using a random partitions to reduce RAM load.",default=1)

    argreader.parser.add_argument('--patch_sim_out_path', type=str,
                                  help="The output path")

    argreader.parser.add_argument('--patch_sim_kertype', type=str,
                                  help="Kernel type. Can be linear, squarred or constant.",default="linear")
    argreader.parser.add_argument('--patch_sim_gram_order', type=int,
                                  help="If 2, computes correlation between feature maps. If 1, computes average of feature maps.",default=1)
    argreader.parser.add_argument('--patch_sim_knn_classes', type=int,
                                  help="The number of classes when using k nearest neighbors.",default=10)

    argreader.parser.add_argument('--data_batch_index', type=int,
                                  help="The index of the batch to process first",default=0)

    argreader.parser.add_argument('--patch_sim_neighsim_nb_iter', type=int,
                                  help="The number of times to apply neighbor similarity averaging ",default=1)
    argreader.parser.add_argument('--patch_sim_neighsim_softmax', type=str2bool,
                                  help="Whether or not to use softmax to weight neighbors",default=False)
    argreader.parser.add_argument('--patch_sim_neighsim_softmax_fact', type=int,
                                  help="Whether or not to use softmax to weight neighbors",default=1)

    argreader.parser.add_argument('--patch_sim_weight_by_neigsim', type=str2bool,
                                  help="To weight a neighbor by similarity",default=True)
    argreader.parser.add_argument('--patch_sim_update_rate_by_cossim', type=str2bool,
                                  help="To define the update rate of the feature using the cossim map.",default=False)
    argreader.parser.add_argument('--patch_sim_neighradius', type=int,
                                  help="The radius of the neighborhood",default=1)
    argreader.parser.add_argument('--patch_sim_neighdilation', type=int,
                                  help="The dilation of the neighborhood",default=1)

    argreader.parser.add_argument('--crop', type=str2bool,
                                  help="To preprocess images by randomly cropping them",default=True)
    argreader.parser.add_argument('--shuffle', type=str2bool,
                                  help="To shuffle dataset",default=True)

    argreader.parser.add_argument('--multi_scale', type=str2bool,
                                  help="To use multi-scale analysis",default=False)
    argreader.parser.add_argument('--second_parse', type=str2bool,
                                  help="To parse the feature as if they were input images",default=False)
    argreader.parser.add_argument('--resnet_multilev', type=str2bool,
                                  help="To use multi-level features with resnet",default=False)

    argreader.parser.add_argument('--second_scale', type=str2bool,
                                  help="To process the image using a second patch size,stride",default=False)
    argreader.parser.add_argument('--patch_size_second', type=int,
                                  help="Patch size for the second processing",default=20)
    argreader.parser.add_argument('--patch_stride_second', type=int,
                                  help="Patch stride for the second processing",default=20)
    argreader.parser.add_argument('--random_farthest', type=str2bool,
                                  help="During neighbor refininement, randomly replace a pixel with its farthest neighbor instead of its closest. ",default=False)
    argreader.parser.add_argument('--random_farthest_prop', type=float,
                                  help="Probability that a pixel is replaced by its closest neighbor.",default=0.7)
    argreader.parser.add_argument('--patch_sim_neiref_neisimavgsize', type=int,
                                  help="The neighborhood size to compare a pixel with when computing similarity.",default=1)
    argreader.parser.add_argument('--patch_sim_neiref_repr_vectors', type=str2bool,
                                  help="To use representative vectors when --patch_sim_neiref_neisimavgsize > 1",default=False)


    argreader.parser.add_argument('--patch_sim_neiref_norm_feat', type=str2bool,
                                  help="To normalize feature before applying neighbor refining",default=False)


    argreader = trainVal.addInitArgs(argreader)
    argreader = trainVal.addOptimArgs(argreader)
    argreader = trainVal.addValArgs(argreader)
    argreader = trainVal.addLossTermArgs(argreader)

    argreader = modelBuilder.addArgs(argreader)
    argreader = load_data.addArgs(argreader)

    # Reading the comand line arg
    argreader.getRemainingArgs()

    args = argreader.args

    #args.second_mod = "pointnet2"
    args.pn_topk = True
    args.pn_topk_euclinorm = False
    args.texture_encoding = True
    args.big_images = True

    net = modelBuilder.netBuilder(args)
    kwargs = {"momentum":args.momentum} if args.optim == "SGD" else {}

    if args.crop:
        transf = None
    else:
        transf = "identity"

    trainLoader, _ = load_data.buildTrainLoader(args,transf=transf,shuffle=args.shuffle)

    data,target =next(iter(trainLoader))
    if args.data_batch_index>0:
        for i in range(args.data_batch_index):
            data,target =next(iter(trainLoader))


    if args.patch_sim:
        with torch.no_grad():

            data = data.cuda() if args.cuda else data

            kwargsNet = {"resnet":args.patch_sim_resnet,"resType":args.patch_sim_restype,"pretr":args.patch_sim_pretr_res,"nbGroup":args.patch_sim_group_nb,\
                        "reluOnSimple":args.patch_sim_relu_on_simple,"chan":args.resnet_chan,"gramOrder":args.patch_sim_gram_order,"useModel":args.patch_sim_usemodel,\
                        "inChan":3,"resnet_multilev":args.resnet_multilev}
            net = buildModule(True,PatchSimCNN,args.cuda,args.multi_gpu,kwargsNet)

            kwargs = {"neigAvgSize":args.patch_sim_neiref_neisimavgsize,"reprVec":args.patch_sim_neiref_repr_vectors}
            cosimMap = buildModule(True,CosimMap,args.cuda,args.multi_gpu,kwargs)

            patch = data.unfold(2, args.patch_size, args.patch_stride).unfold(3, args.patch_size, args.patch_stride).permute(0,2,3,1,4,5)
            origPatchSize = patch.size()
            patch = patch.reshape(patch.size(0)*patch.size(1)*patch.size(2),patch.size(3),patch.size(4),patch.size(5))

            kwargsNeiSim = {"cuda":args.cuda,"groupNb":args.patch_sim_group_nb,"nbIter":args.patch_sim_neighsim_nb_iter,\
                        "softmax":args.patch_sim_neighsim_softmax,"softmax_fact":args.patch_sim_neighsim_softmax_fact,\
                        "weightByNeigSim":args.patch_sim_weight_by_neigsim,"updateRateByCossim":args.patch_sim_update_rate_by_cossim,"neighRadius":args.patch_sim_neighradius,\
                        "neighDilation":args.patch_sim_neighdilation,"random_farthest":args.random_farthest,"random_farthest_prop":args.random_farthest_prop,\
                        "neigAvgSize":args.patch_sim_neiref_neisimavgsize,"reprVec":args.patch_sim_neiref_repr_vectors,"normFeat":args.patch_sim_neiref_norm_feat}
            neighSimMod = buildModule(True,NeighSim,args.cuda,args.multi_gpu,kwargsNeiSim)

            representativeVectorsMod = buildModule(True,RepresentativeVectors,args.cuda,args.multi_gpu,{})
            computeTotalSimMod = buildModule(True,ComputeTotalSim,args.cuda,args.multi_gpu,{})

            print("Start !")
            start = time.time()

            def textLimit(patch,secondParse=False):

                kwargsNet["inChan"] = patch.size(1)
                patchMod = buildModule(True,PatchSimCNN,args.cuda,args.multi_gpu,kwargsNet)
                gram_mat = patchMod(patch)
                gram_mat = gram_mat.view(data.size(0),-1,args.patch_sim_group_nb,gram_mat.size(2))
                gram_mat = gram_mat.permute(0,2,1,3)
                distMapAgreg,feat = cosimMap(gram_mat)
                neighSim,refFeatList = neighSimMod(feat)
                if args.multi_scale:
                    multiScaleKer = torch.ones((feat.size(1),1,21,21)).to(feat.device)
                    multiScaleKer /= (feat.size(-2)*feat.size(-1))
                    multiScaleFeat = F.conv2d(feat,multiScaleKer,groups=feat.size(1),padding=21//2)

                    kwargsNeiSim["neighDilation"] = 5
                    neighSimMod_multiscale = buildModule(True,NeighSim,args.cuda,args.multi_gpu,kwargsNeiSim)
                    #neighSim_small,refFeat_small = neighSimMod_multiscale(multiScaleFeat)
                    refFeat_small = multiScaleFeat
                    neighSim_small = simMap = computeTotalSim(multiScaleFeat,1)

                else:
                    neighSim_small,refFeat_small = None,None

                return feat,distMapAgreg,neighSim,neighSim_small,refFeatList,refFeat_small

            feat,distMapAgreg,neighSim,neighSim_small,refFeatList,refFeat_small = textLimit(patch)

            #if args.patch_sim_neiref_neisimavgsize > 1:
                #refFeatRepList = []
                #neighSimRepList = []
                #for i in range(len(refFeatList)):
                #    featRepr = representativeVectorsMod(refFeatList[i],args.patch_sim_neiref_neisimavgsize)
                #    refFeatRepList.append(featRepr)
                #    neighSimRepList.append(computeTotalSimMod(featRepr,args.patch_sim_neiref_neisimavgsize))
            #else:
            #    refFeatRepList = None
            #    neighSimRepList = None

            if args.second_parse:
                feat_patch = refFeatList[-1].unfold(2, args.patch_size_second, args.patch_stride_second).unfold(3, args.patch_size_second,args.patch_stride_second).permute(0,2,3,1,4,5)
                origFeatPatchSize = feat_patch.size()
                feat_patch = feat_patch.reshape(feat_patch.size(0)*feat_patch.size(1)*feat_patch.size(2),feat_patch.size(3),feat_patch.size(4),feat_patch.size(5))
                _,featDistMapAgreg,featNeighSim,featNeighSim_small,featRefFeat,_ = textLimit(feat_patch)
            else:
                featNeighSim,featNeighSim_small,featRefFeat = None,None,None

            if args.second_scale:
                patch_second = data.unfold(2, args.patch_size_second, args.patch_stride_second).unfold(3, args.patch_size_second,args.patch_stride_second).permute(0,2,3,1,4,5)
                origSecondPatchSize = patch_second.size()
                patch_second = patch_second.reshape(patch_second.size(0)*patch_second.size(1)*patch_second.size(2),patch_second.size(3),patch_second.size(4),patch_second.size(5))
                secondDistMapAgreg,secondNeighSim,secondNeighSim_small,secondRefFeat,_ = textLimit(patch_second)
            else:
                secondDistMapAgreg,secondNeighSim,secondRefFeat = None,None,None
            print("End ",time.time()-start)

            if not os.path.exists("../vis/{}/".format(args.exp_id)):
                os.makedirs("../vis/{}/".format(args.exp_id))
            if not os.path.exists("../results/{}/".format(args.exp_id)):
                os.makedirs("../results/{}/".format(args.exp_id))

            patch = patch.reshape(origPatchSize).detach().cpu().numpy()

            if not os.path.exists(os.path.join(args.patch_sim_out_path,"imgs")):
                os.makedirs(os.path.join(args.patch_sim_out_path,"imgs"))
            if not os.path.exists(os.path.join(args.patch_sim_out_path,"edges")):
                os.makedirs(os.path.join(args.patch_sim_out_path,"edges"))
            if not os.path.exists(os.path.join(args.patch_sim_out_path,"sobel")):
                os.makedirs(os.path.join(args.patch_sim_out_path,"sobel"))
            if not os.path.exists(os.path.join(args.patch_sim_out_path,"simMaps")):
                os.makedirs(os.path.join(args.patch_sim_out_path,"simMaps"))
            if not os.path.exists(os.path.join(args.patch_sim_out_path,"simMaps",args.model_id)):
                os.makedirs(os.path.join(args.patch_sim_out_path,"simMaps",args.model_id))

            if not secondNeighSim is None:
                for i in range(len(neighSim)):
                    neighSim[i] = (neighSim[i]-neighSim[i].min())/(neighSim[i].max()-neighSim[i].min())
                    secondNeighSim[i] = (secondNeighSim[i]-secondNeighSim[i].min())/(secondNeighSim[i].max()-secondNeighSim[i].min())

                firstAndSecondNeighSim_mean = 0.5*neighSim+0.5*torch.nn.functional.interpolate(secondNeighSim, size=(neighSim.size(-2),neighSim.size(-1)),mode="bilinear",align_corners=False)
                firstAndSecondNeighSim_max = torch.min(neighSim,torch.nn.functional.interpolate(secondNeighSim, size=(neighSim.size(-2),neighSim.size(-1)),mode="bilinear",align_corners=False))
                firstAndSecondNeighSim_min = torch.max(neighSim,torch.nn.functional.interpolate(secondNeighSim, size=(neighSim.size(-2),neighSim.size(-1)),mode="bilinear",align_corners=False))

            neighSim = neighSim.detach().cpu().numpy()

            #if not neighSimRepList is None:
            #    for i in range(len(neighSimRepList)):
            #        neighSimRepList[i] = neighSimRepList[i].detach().cpu().numpy()

            if not neighSim_small is None:
                neighSim_small = neighSim_small.detach().cpu().numpy()

            if not featNeighSim is None:
                featNeighSim = featNeighSim.detach().cpu().numpy()

            if not secondNeighSim is None:
                secondNeighSim = secondNeighSim.detach().cpu().numpy()
                firstAndSecondNeighSim_mean = firstAndSecondNeighSim_mean.detach().cpu().numpy()
                firstAndSecondNeighSim_max = firstAndSecondNeighSim_max.detach().cpu().numpy()
                firstAndSecondNeighSim_min = firstAndSecondNeighSim_min.detach().cpu().numpy()

            for i,img in enumerate(data):

                if not os.path.exists(os.path.join(args.patch_sim_out_path,"simMaps",args.model_id,"{}".format(i+args.data_batch_index*args.batch_size))):
                    os.makedirs(os.path.join(args.patch_sim_out_path,"simMaps",args.model_id,"{}".format(i+args.data_batch_index*args.batch_size)))

                img = (img-img.min())/(img.max()-img.min())

                plotImg(img.detach().cpu().permute(1,2,0).numpy(),os.path.join(args.patch_sim_out_path,"imgs","{}.png".format(i+args.data_batch_index*args.batch_size)))

                resizedImg = resize(img.detach().cpu().permute(1,2,0).numpy().mean(axis=-1), (100,100),anti_aliasing=True,mode="constant",order=0)*255
                edges = ~skimage.feature.canny(resizedImg,sigma=3)
                plotImg(edges,os.path.join(args.patch_sim_out_path,"edges","{}.png".format(i+args.data_batch_index*args.batch_size)))

                sobel = sobelFunc(img.detach().cpu().permute(1,2,0).numpy().mean(axis=-1))
                plotImg(sobel,os.path.join(args.patch_sim_out_path,"sobel","{}.png".format(i+args.data_batch_index*args.batch_size)))

                sobel = sobelFunc(img.detach().cpu().permute(1,2,0).numpy().mean(axis=-1))
                minima,_,_ = computeMinima(sobel)
                plotImg(255-((255-sobel)*minima),os.path.join(args.patch_sim_out_path,"sobel","nms-{}.png".format(i+args.data_batch_index*args.batch_size)))

                topk(sobel,minima,os.path.join(args.patch_sim_out_path,"sobel"),"{}".format(i+args.data_batch_index*args.batch_size))

                simMapPath = os.path.join(args.patch_sim_out_path,"simMaps",args.model_id,str(i+args.data_batch_index*args.batch_size))

                writeAllImg(distMapAgreg,neighSim,i,simMapPath,"sparseNeighSim_step0")
                dimRedList(refFeatList,simMapPath,i,"neiRef")

                #if not refFeatRepList is None:
                #    dimRedList(refFeatRepList,simMapPath,i,"neiRef_repr")
                #    for j in range(len(neighSimRepList)):
                #        pathPNG = os.path.join(simMapPath,"sparseNeighSim_step{}_repr".format(len(neighSimRepList)-1-j))
                #        plotImg(neighSimRepList[j][i][0],pathPNG,cmap="gray")

                if not neighSim_small is None:
                    writeAllImg(None,neighSim_small,i,simMapPath,"sparseNeighSim_step0_x2")
                    dimRed(refFeat_small,simMapPath,i,"neiRef_small")

                if not featNeighSim is None:
                    writeAllImg(featDistMapAgreg,featNeighSim,i,simMapPath,"sparseNeighSim_step0_feat")

                if not secondNeighSim is None:
                    writeAllImg(secondDistMapAgreg,secondNeighSim,i,simMapPath,"sparseNeighSim_step0_second")
                    writeAllImg(None,firstAndSecondNeighSim_mean,i,simMapPath,"sparseNeighSim_step0_firstAndSecond_mean")
                    writeAllImg(None,firstAndSecondNeighSim_min,i,simMapPath,"sparseNeighSim_step0_firstAndSecond_min")
                    writeAllImg(None,firstAndSecondNeighSim_max,i,simMapPath,"sparseNeighSim_step0_firstAndSecond_max")

def writeAllImg(distMapAgreg,neighSim,i,simMapPath,name):
    if not distMapAgreg is None:
        plotImg(distMapAgreg[i][0][1:-1,1:-1].detach().cpu().numpy(),os.path.join(simMapPath,name+"_agr.png"),'gray')

    neighSim[i] = 255*(neighSim[i]-neighSim[i].min())/(neighSim[i].max()-neighSim[i].min())
    for j in range(len(neighSim[i])):
        pathPNG = os.path.join(simMapPath,name.replace("step0","step{}".format(len(neighSim[i])-1-j)))
        plotImg(neighSim[i][j],pathPNG,cmap="gray")

    #j = len(neighSim[i])-1
    #minima,minimaV,minimaH = computeMinima(neighSim[i][-1])
    #topk(neighSim[i][j],minima,simMapPath,name)

def dimRed(refFeat,simMapPath,i,name):

    #refFeat_emb = umap.UMAP().fit_transform(refFeat[i].view(refFeat[i].size(0),-1).permute(1,0).cpu().detach().numpy())

    #plt.figure()
    #plt.scatter(refFeat_emb[:,0],refFeat_emb[:,1],alpha=0.1,edgecolors=None)
    #plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    #plt.savefig(os.path.join(simMapPath,"{}_tsne.png".format(name)))
    #plt.close()

    refFeat_emb = umap.UMAP(n_components=3).fit_transform(refFeat[i].view(refFeat[i].size(0),-1).permute(1,0).cpu().detach().numpy())
    refFeat_emb = refFeat_emb.reshape((refFeat[i].size(1),refFeat[i].size(2),3))
    refFeat_emb = (refFeat_emb-refFeat_emb.min())/(refFeat_emb.max()-refFeat_emb.min())
    plt.figure(figsize=(10,10))
    plt.imshow(refFeat_emb)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(os.path.join(simMapPath,"{}_tsne_img.png".format(name)))
    plt.close()


def dimRedList(refFeatList,simMapPath,i,name):

    #refFeat_emb = umap.UMAP().fit_transform(refFeat[i].view(refFeat[i].size(0),-1).permute(1,0).cpu().detach().numpy())

    #plt.figure()
    #plt.scatter(refFeat_emb[:,0],refFeat_emb[:,1],alpha=0.1,edgecolors=None)
    #plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    #plt.savefig(os.path.join(simMapPath,"{}_tsne.png".format(name)))
    #plt.close()

    #if not os.path.exists(os.path.join(simMapPath,"{}_{}_tsne_img.png".format(name,0))):
    refFeatList_ex = [refFeatList[j][i] for j in range(len(refFeatList))]
    for j in range(len(refFeatList_ex)):
        refFeatList_ex[j] = refFeatList_ex[j].unsqueeze(0)
    refFeatList_ex = torch.cat(refFeatList_ex,dim=0)

    origShape = refFeatList_ex.size()
    refFeatList_ex = refFeatList_ex.permute(1,0,2,3)

    refFeat_emb = umap.UMAP(n_components=3).fit_transform(refFeatList_ex.reshape(refFeatList_ex.size(0),-1).permute(1,0).cpu().detach().numpy())

    refFeat_emb = refFeat_emb.reshape((origShape[0],origShape[2],origShape[3],3))
    refFeat_emb = (refFeat_emb-refFeat_emb.min())/(refFeat_emb.max()-refFeat_emb.min())

    for j in range(len(refFeat_emb)):
        plt.figure(figsize=(10,10))
        plt.imshow(refFeat_emb[j])
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(os.path.join(simMapPath,"{}_{}_tsne_img.png".format(name,j)))
        plt.close()

            #norm = torch.sqrt(torch.pow(refFeatList_ex[j],2).sum(dim=0)).cpu().detach().numpy()
            #plt.figure(figsize=(10,10))
            #plt.imshow(norm)
            #plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            #plt.savefig(os.path.join(simMapPath,"{}_{}_norm.png".format(name,j)))
            #plt.close()
            #norm = (norm-norm.min())/(norm.max()-norm.min())

            #emb_norm = norm[:,:,np.newaxis]*refFeat_emb[j]
            #plt.figure(figsize=(10,10))
            #plt.imshow(emb_norm)
            #plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
            #plt.savefig(os.path.join(simMapPath,"{}_{}_tsne_img_norm.png".format(name,j)))
            #plt.close()

def normList(refFeatListFull,simMapPath,i,name):

    for j in range(len(refFeatListFull)):
        norm = torch.sqrt(torch.pow(refFeatListFull[j][i],2).sum(dim=0))
        plt.figure(figsize=(10,10))
        plt.imshow(norm.cpu().detach().numpy())
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        plt.savefig(os.path.join(simMapPath,"{}_{}_norm.png".format(name,j)))
        plt.close()

def topk(img,minima,folder,fileName):

    pathPNG = os.path.join(folder,"nms-{}.png".format(fileName))
    plotImg(255-((255-img)*minima),pathPNG)

    for k in [512,1024,2048]:
        flatInds = np.argsort(img.reshape(-1))[:k]
        abs, ord = (flatInds % img.shape[-1], flatInds // img.shape[-1])
        img_cpy = np.zeros_like(img)
        img_cpy[:] = 255
        img_cpy[ord,abs] = img[ord,abs]
        pathPNG = os.path.join(folder,"{}-{}.png".format(k,fileName))
        plotImg(img_cpy,pathPNG,cmap="gray")

        #Topk and then NMS
        pathPNG = os.path.join(folder,"{}-nms-{}.png".format(k,fileName))
        plotImg(255-((255-img_cpy)*minima),pathPNG,cmap="gray")

        #NMS and then Topk
        flatInds = np.argsort((255-((255-img)*minima)).reshape(-1))[:k]
        abs, ord = (flatInds % img.shape[-1], flatInds // img.shape[-1])
        img_cpy = np.zeros_like(img)
        img_cpy[:] = 255

        img_cpy[ord,abs] = img[ord,abs]

        pathPNG = os.path.join(folder,"nms-{}-{}.png".format(k,fileName))
        plotImg(img_cpy,pathPNG,cmap="gray")

def enhanceBlack(arr):
    arr = (arr-arr.min())/(arr.max()-arr.min())
    return 255*(arr*arr*arr*arr)

def botk(img,maxima,folder,fileName):

    for k in [512,1024,2048]:
        flatInds = np.argsort((img*maxima).reshape(-1))[-2*k:]

        abs, ord = (flatInds % img.shape[-1], flatInds // img.shape[-1])
        abs,ord = torch.tensor(abs).unsqueeze(1), torch.tensor(ord).unsqueeze(1)
        x = torch.cat((abs,ord),dim=1)
        inds = torch_geometric.nn.fps(x.float(), ratio=k/(len(img.reshape(-1))-k))
        abs,ord = abs[inds].numpy(),ord[inds].numpy()

        img_cpy = np.zeros_like(img)
        img_cpy[:] = 0
        img_cpy[ord,abs] = img[ord,abs]
        pathPNG = os.path.join(folder,"nms-{}-{}.png".format(k,fileName))
        plotImg(img_cpy,pathPNG,cmap="gray")

def computeMinima(img):
    neiSimMinV = filters.minimum_filter(img, (3,1))
    neiSimMinH = filters.minimum_filter(img, (1,3))
    minima = np.logical_or((img == neiSimMinV),(img == neiSimMinH))
    minimaV = (img == neiSimMinV)
    minimaH = (img == neiSimMinH)
    return minima,minimaV,minimaH

def computeMaxima(img):
    neiSimMaxV = filters.maximum_filter(img, (3,1))
    neiSimMaxH = filters.maximum_filter(img, (1,3))
    minima = np.logical_or((img == neiSimMaxV),(img == neiSimMaxH))
    minimaV = (img == neiSimMaxV)
    minimaH = (img == neiSimMaxH)
    return minima,minimaV,minimaH

def sobelFunc(img):

    img = (255*(img-img.min())/(img.max()-img.min()))
    #img = resize(img, (100,100),anti_aliasing=True,mode="constant",order=3)*255

    img = img.astype('int32')
    dx = ndimage.sobel(img, 0)  # horizontal derivative
    dy = ndimage.sobel(img, 1)  # vertical derivative
    mag = np.hypot(dx, dy)  # magnitude
    #mag = dx.astype("float64")
    mag *= 255.0 / np.max(mag)  # normalize (Q&D)
    mag= mag.astype("uint8")
    return 255-mag

def plotImg(img,path,cmap="gray"):
    plt.figure(figsize=(10,10))
    if cmap is None:
        plt.imshow(img)
    else:
        plt.imshow(img,cmap)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(path)
    plt.close()

class RepresentativeVectors(torch.nn.Module):
    def __init__(self):
        super(RepresentativeVectors,self).__init__()
    def forward(self,features,patch_sim_neiref_neisimavgsize):
        return representativeVectors(features,patch_sim_neiref_neisimavgsize)

class ComputeTotalSim(torch.nn.Module):
    def __init__(self):
        super(ComputeTotalSim,self).__init__()
    def forward(self,features,patch_sim_neiref_neisimavgsize):
        return computeTotalSim(features,1,patch_sim_neiref_neisimavgsize,False)

class GramDistMap(torch.nn.Module):
    def __init__(self,full_mode,kerSize,parralelMode):
        super(GramDistMap,self).__init__()
        self.full_mode = full_mode
        self.kerSize = kerSize
        self.parralelMode = parralelMode
    def forward(self,gramBatch,patchPerRow):

        if not self.parralelMode:
            distMat_mean = None
            for i in range(gramBatch.size(1)):
                gram = gramBatch[:,i]
                gram = gram.reshape(gram.size(0),gram.size(1),gram.size(2)*gram.size(3))
                gramNorm = torch.sqrt(torch.pow(gram*gram,2).sum(dim=-1))

                if not self.full_mode:
                    distMat = -torch.ones((gram.size(0),gram.size(1),gram.size(1))).to(gram.device)
                    for i in range(distMat.size(1)):
                        xi,yi = i%patchPerRow,i//patchPerRow
                        for j in range(distMat.size(2)):
                            xj,yj = j%patchPerRow,j//patchPerRow
                            if np.abs(xi-xj)+np.abs(yi-yj) < self.kerSize:
                                distMat[:,i,j] = torch.pow(gram[:,i]-gram[:,j],2).sum(dim=-1)/(gramNorm[:,i]*gramNorm[:,j])
                else:
                    distMat = torch.zeros((gram.size(0),gram.size(1),gram.size(1))).to(gram.device)
                    for i in range(distMat.size(1)):
                        for j in range(distMat.size(2)):
                            distMat[:,i,j] = torch.pow(gram[:,i]-gram[:,j],2).sum(dim=-1)/(gramNorm[:,i]*gramNorm[:,j])

                if distMat_mean is None:
                    distMat_mean = distMat
                else:
                    distMat_mean += distMat

            distMat_mean /= len(gramList)

        else:
            gram = gramBatch
            #gram = gram.reshape(gram.size(0),gram.size(1),gram.size(2),gram.size(3)*gram.size(4))
            gramNorm = torch.sqrt(torch.pow(gram*gram,2).sum(dim=-1))
            gramX = gram.unsqueeze(2)
            gramY = gram.unsqueeze(3)
            gramXNorm = gramNorm.unsqueeze(2)
            gramYNorm = gramNorm.unsqueeze(3)
            distMat = (torch.pow(gramX-gramY,2).sum(dim=-1)/(gramXNorm*gramYNorm))
            distMat_mean = distMat.mean(dim=1)

        return distMat_mean

class PatchSimCNN(torch.nn.Module):
    def __init__(self,resnet,resType,pretr,nbGroup,reluOnSimple,gramOrder,useModel,resnet_multilev,**kwargs):
        super(PatchSimCNN,self).__init__()

        self.resnet = resnet
        if not resnet:
            self.filterSizes = [3,5,7,11,15,23,37,55]
            self.layers = torch.nn.ModuleList()

            channelPerFilterSize = kwargs["chan"]//8
            for filterSize in self.filterSizes:
                if reluOnSimple:
                    layer = torch.nn.Sequential(torch.nn.Conv2d(kwargs["inChan"],channelPerFilterSize,filterSize,padding=(filterSize-1)//2),torch.nn.ReLU())
                else:
                    layer = torch.nn.Conv2d(kwargs["inChan"],channelPerFilterSize,filterSize,padding=(filterSize-1)//2)
                self.layers.append(layer)
        else:
            if resnet_multilev:
                kwargs["chan"] = [kwargs["chan"]//4 for _ in range(4)]
                self.featMod = modelBuilder.buildFeatModel(resType, pretr, True, False,**kwargs)
            else:
                kwargs["chan"] = kwargs["chan"]//2
                self.featMod = modelBuilder.buildFeatModel(resType, pretr, True, False,**kwargs)

        self.nbGroup = nbGroup
        self.gramOrder = gramOrder
        self.useModel = useModel
        self.resnet_multilev = resnet_multilev
    def forward(self,x):

        if self.useModel:
            if not self.resnet:
                featList = []
                for i,layer in enumerate(self.layers):
                    layerFeat = layer(x)
                    featList.append(layerFeat)

                featVolume = torch.cat(featList,dim=1)
                featVolume = featVolume[:,torch.randperm(featVolume.size(1))]
            else:
                if self.resnet_multilev:
                    featVolume = torch.cat([self.featMod(x)["layerFeat"][i] for i in range(1,5)],dim=1)
                else:
                    featVolume = self.featMod(x)["x"]
        else:
            featVolume = torch.nn.functional.adaptive_avg_pool2d(x,(3,3))

        origFeatSize = featVolume.size()
        featVolume = featVolume.unfold(1, featVolume.size(1)//self.nbGroup, featVolume.size(1)//self.nbGroup).permute(0,1,4,2,3)
        featVolume = featVolume.view(featVolume.size(0)*featVolume.size(1),featVolume.size(2),featVolume.size(3),featVolume.size(4))

        gramMat = graham(featVolume,self.gramOrder)
        #gramMat = gramMat.view(origFeatSize[0],self.nbGroup,origFeatSize[1]//self.nbGroup,origFeatSize[1]//self.nbGroup)
        gramMat = gramMat.view(origFeatSize[0],self.nbGroup,gramMat.size(-1))

        return gramMat

class CosimMap(torch.nn.Module):
    def __init__(self,neigAvgSize,reprVec):
        super(CosimMap,self).__init__()
        self.neigAvgSize = neigAvgSize
        self.reprVec = reprVec

    def forward(self,x):

        origSize = x.size()

        #N x NbGroup x nbPatch x C
        x = x.reshape(x.size(0)*x.size(1),x.size(2),x.size(3))
        # (NxNbGroup) x nbPatch x C
        x = x.permute(0,2,1)
        # (NxNbGroup) x C x nbPatch
        x = x.unsqueeze(-1)
        # (NxNbGroup) x C x nbPatch x 1
        feat = x.reshape(x.size(0),x.size(1),int(np.sqrt(x.size(2))),int(np.sqrt(x.size(2))))
        # (NxNbGroup) x C x sqrt(nbPatch) x sqrt(nbPatch)
        x = computeTotalSim(feat,1,self.neigAvgSize,self.reprVec).unsqueeze(1)
        # (NxNbGroup) x 1 x 1 x sqrt(nbPatch) x sqrt(nbPatch)
        x = x.reshape(origSize[0],origSize[1],x.size(2),x.size(3),x.size(4))
        # N x NbGroup x 1 x sqrt(nbPatch) x sqrt(nbPatch)
        x = x.mean(dim=1)
        # N x 1 x sqrt(nbPatch) x sqrt(nbPatch)
        return x,feat

def computeNeighborsCoord(neighRadius):

    coord = torch.arange(neighRadius*2+1)-neighRadius

    y = coord.unsqueeze(1).expand(coord.size(0),coord.size(0)).unsqueeze(-1)
    x = coord.unsqueeze(0).expand(coord.size(0),coord.size(0)).unsqueeze(-1)

    coord = torch.cat((x,y),dim=-1).view(coord.size(0)*coord.size(0),2)
    coord = coord[~((coord[:,0] == 0)*(coord[:,1] == 0))]

    return coord


def compositeShiftFeat(coord,features):

    if coord[0] != 0:
        if coord[0] > 0:
            shiftFeatV,shiftMaskV = shiftFeat("top",features,coord[0])
        else:
            shiftFeatV,shiftMaskV = shiftFeat("bot",features,-coord[0])
    else:
        shiftFeatV,shiftMaskV = shiftFeat("none",features)

    if coord[1] != 0:
        if coord[1] > 0:
            shiftFeatH,shiftMaskH = shiftFeat("right",shiftFeatV,coord[1])
        else:
            shiftFeatH,shiftMaskH = shiftFeat("left",shiftFeatV,-coord[1])
    else:
        shiftFeatH,shiftMaskH = shiftFeat("none",shiftFeatV)

    return shiftFeatH,shiftMaskH*shiftMaskV

def shiftFeat(where,features,dilation=1,neigAvgSize=1,reprVec=False):

    mask = torch.ones_like(features)

    if neigAvgSize > 1:
        dilation = neigAvgSize//2

    if where=="left":
        #x,y = 0,1
        padd = features[:,:,:,-1:].expand(-1,-1,-1,dilation)
        paddMask = torch.zeros((features.size(0),features.size(1),features.size(2),dilation)).to(features.device)+0.0001
        featuresShift = torch.cat((features[:,:,:,dilation:],padd),dim=-1)
        maskShift = torch.cat((mask[:,:,:,dilation:],paddMask),dim=-1)
    elif where=="right":
        #x,y= 2,1
        padd = features[:,:,:,:1].expand(-1,-1,-1,dilation)
        paddMask = torch.zeros((features.size(0),features.size(1),features.size(2),dilation)).to(features.device)+0.0001
        featuresShift = torch.cat((padd,features[:,:,:,:-dilation]),dim=-1)
        maskShift = torch.cat((paddMask,mask[:,:,:,:-dilation]),dim=-1)
    elif where=="bot":
        #x,y = 1,0
        padd = features[:,:,:1].expand(-1,-1,dilation,-1)
        paddMask = torch.zeros((features.size(0),features.size(1),dilation,features.size(3))).to(features.device)+0.0001
        featuresShift = torch.cat((padd,features[:,:,:-dilation,:]),dim=-2)
        maskShift = torch.cat((paddMask,mask[:,:,:-dilation,:]),dim=-2)
    elif where=="top":
        #x,y = 1,2
        padd = features[:,:,-1:].expand(-1,-1,dilation,-1)
        paddMask = torch.zeros((features.size(0),features.size(1),dilation,features.size(3))).to(features.device)+0.0001
        featuresShift = torch.cat((features[:,:,dilation:,:],padd),dim=-2)
        maskShift = torch.cat((mask[:,:,dilation:,:],paddMask),dim=-2)
    elif where=="none":
        featuresShift = features
        maskShift = mask
    else:
        raise ValueError("Unkown position")

    if neigAvgSize > 1:
        if reprVec:
            featuresShift = representativeVectors(featuresShift,neigAvgSize)
        else:
            featuresShift = F.conv2d(featuresShift,torch.ones(featuresShift.size(1),1,neigAvgSize,neigAvgSize).to(featuresShift.device)/(neigAvgSize*neigAvgSize),groups=featuresShift.size(1),padding=neigAvgSize//2)

        #print(featuresShift.min().item(),featuresShift.mean().item(),featuresShift.max().item())
        #sys.exit(0)

    maskShift = maskShift.mean(dim=1,keepdim=True)
    return featuresShift,maskShift

def representativeVectors(x,kerSize=5):

    #x = torch.ones(1,1,kerSize,kerSize)
    #x = torch.normal(torch.ones(1,1,kerSize,kerSize), 0.001*torch.ones(1,1,kerSize,kerSize))
    #x[0,0,0,0] = -1
    origShape = x.size()
    #print(x.size())
    patch = F.unfold(x,kerSize,padding=kerSize//2)+0.00001
    #print("unfold",patch.size())
    patch = patch.permute(0,2,1)
    #print("permute",patch.size())
    patch = patch.reshape(origShape[0],origShape[2],origShape[3],origShape[1],kerSize,kerSize)
    #print("reshape",patch.size())
    patch = patch.reshape(patch.size(0)*patch.size(1)*patch.size(2),patch.size(3),patch.size(4)*patch.size(5))
    #print("reshape",patch.size())
    patch = patch.permute(0,2,1)
    #print("permute",patch.size())
    #print("patch size before norm",patch.size())

    patchNorm = patch/torch.sqrt(torch.pow(patch,2).sum(dim=-1,keepdim=True))
    sim = (patchNorm*patchNorm[:,patch.size(1)//2:patch.size(1)//2+1]).sum(dim=-1,keepdim=True)
    #print(patchNorm.size(),patchNorm[:,patch.size(1)//2:patch.size(1)//2+1].size(),(patchNorm*patchNorm[:,patch.size(1)//2:patch.size(1)//2+1]).size())
    #print("sim",sim.max(),sim.min(),sim.mean())
    #sim = (sim+1)/2
    sim = torch.softmax(50*sim,dim=1)
    #sim = sim/sim.sum(dim=1,keepdim=True)
    #print("sim",sim.max(),sim.min(),sim.mean())

    reshapedPatch = patch.reshape(origShape[0],origShape[-2],origShape[-1],patch.size(-2),patch.size(-1))

    #print(patch.size(),sim.size())
    #print("Computing the ponderated average")
    reprVec = (patch*sim).sum(dim=1,keepdim=True)
    #print(reprVec.size())
    reprVec = reprVec.reshape(origShape[0],origShape[2],origShape[3],reprVec.size(-2),reprVec.size(-1))
    #print(reprVec.size())
    reprVec = reprVec.squeeze(dim=-2)
    #print(reprVec.size())
    reprVec = reprVec.permute(0,3,1,2)
    #print(x.size(),reprVec.size())

    #sys.exit(0)

    reprVecNoWei = patch.mean(dim=1,keepdim=True)
    reprVecNoWei = reprVecNoWei.reshape(origShape[0],origShape[2],origShape[3],reprVecNoWei.size(-2),reprVecNoWei.size(-1))
    reprVecNoWei = reprVecNoWei.squeeze(dim=-2)
    reprVecNoWei = reprVecNoWei.permute(0,3,1,2)

    '''
    print(reprVecNoWei.min(dim=-1)[0].min(dim=-1)[0].min(dim=-1)[0])
    print(reshapedPatch.min(dim=-1)[0].min(dim=-1)[0].min(dim=-1)[0].min(dim=-1)[0])
    print(reprVec.min(dim=-1)[0].min(dim=-1)[0].min(dim=-1)[0])
    print(x.min(dim=-1)[0].min(dim=-1)[0].min(dim=-1)[0])

    print(reprVecNoWei.max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0])
    print(reshapedPatch.max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0])
    print(reprVec.max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0])
    print(x.max(dim=-1)[0].max(dim=-1)[0].max(dim=-1)[0])

    sys.exit(0)
    '''

    return reprVec

def applyDiffKer_CosSimi(direction,features,dilation=1,neigAvgSize=1,reprVec=False):
    origFeatSize = features.size()
    featNb = origFeatSize[1]

    if type(direction) is str:
        if direction == "horizontal":
            featuresShift1,maskShift1 = shiftFeat("right",features,dilation,neigAvgSize,reprVec)
            featuresShift2,maskShift2 = shiftFeat("left",features,dilation,neigAvgSize,reprVec)
        elif direction == "vertical":
            featuresShift1,maskShift1 = shiftFeat("top",features,dilation,neigAvgSize,reprVec)
            featuresShift2,maskShift2 = shiftFeat("bot",features,dilation,neigAvgSize,reprVec)
        elif direction == "top":
            featuresShift1,maskShift1 = shiftFeat("top",features,dilation,neigAvgSize,reprVec)
            featuresShift2,maskShift2 = shiftFeat("none",features)
        elif direction == "bot":
            featuresShift1,maskShift1 = shiftFeat("bot",features,dilation,neigAvgSize,reprVec)
            featuresShift2,maskShift2 = shiftFeat("none",features)
        elif direction == "left":
            featuresShift1,maskShift1 = shiftFeat("left",features,dilation,neigAvgSize,reprVec)
            featuresShift2,maskShift2 = shiftFeat("none",features)
        elif direction == "right":
            featuresShift1,maskShift1 = shiftFeat("right",features,dilation,neigAvgSize,reprVec)
            featuresShift2,maskShift2 = shiftFeat("none",features)
        elif direction == "none":
            featuresShift1,maskShift1 = shiftFeat("none",features)
            featuresShift2,maskShift2 = shiftFeat("none",features)
        else:
            raise ValueError("Unknown direction : ",direction)
    else:
        featuresShift1,maskShift1 = compositeShiftFeat(direction,features)
        featuresShift2,maskShift2 = shiftFeat("none",features)

    #if direction == "horizontal":
    #    print(neigAvgSize,reprVec)
    #    print(featuresShift1.min().item(),featuresShift1.mean().item(),featuresShift1.max().item())
    #    print(featuresShift2.min().item(),featuresShift2.mean().item(),featuresShift2.max().item())
    #    sys.exit(0)

    sim = (featuresShift1*featuresShift2*maskShift1*maskShift2).sum(dim=1,keepdim=True)
    sim /= torch.sqrt(torch.pow(maskShift1*featuresShift1,2).sum(dim=1,keepdim=True))*torch.sqrt(torch.pow(maskShift2*featuresShift2,2).sum(dim=1,keepdim=True))

    return sim,featuresShift1,featuresShift2,maskShift1,maskShift2

def computeTotalSim(features,dilation,neigAvgSize=1,reprVec=False):
    horizDiff,_,_,_,_ = applyDiffKer_CosSimi("horizontal",features,dilation,neigAvgSize,reprVec)
    vertiDiff,_,_,_,_ = applyDiffKer_CosSimi("vertical",features,dilation,neigAvgSize,reprVec)
    totalDiff = (horizDiff + vertiDiff)/2
    return totalDiff


class NeighSim(torch.nn.Module):
    def __init__(self,cuda,groupNb,nbIter,softmax,softmax_fact,weightByNeigSim,updateRateByCossim,neighRadius,neighDilation,random_farthest,random_farthest_prop,\
                        neigAvgSize,reprVec,normFeat):
        super(NeighSim,self).__init__()

        self.directions = computeNeighborsCoord(neighRadius)

        self.sumKer = torch.ones((1,len(self.directions),1,1))
        self.sumKer = self.sumKer.cuda() if cuda else self.sumKer
        self.groupNb = groupNb
        self.nbIter = nbIter
        self.softmax = softmax
        self.softmax_fact = softmax_fact

        self.weightByNeigSim = weightByNeigSim
        self.updateRateByCossim = updateRateByCossim

        if not self.weightByNeigSim and self.softmax:
            raise ValueError("Can't have weightByNeigSim=False and softmax=True")

        self.random_farthest = random_farthest
        if self.random_farthest:
            self.distr = torch.distributions.bernoulli.Bernoulli(probs=torch.tensor([random_farthest_prop]))
            self.neighDilation = neighDilation
        else:
            self.distr=None
            self.neighDilation = None

        self.neigAvgSize = neigAvgSize
        self.reprVec = reprVec
        self.normFeat = normFeat

    def computeSimAndShiftFeat(self,features,dilation=1):
        allSim = []
        allFeatShift = []
        for direction in self.directions:
            if self.weightByNeigSim:
                sim,featuresShift1,_,maskShift1,_ = applyDiffKer_CosSimi(direction*dilation,features,1,self.neigAvgSize,self.reprVec)
                allSim.append(sim*maskShift1)
            else:
                featuresShift1,maskShift1 = shiftFeat(direction,features,1)
                allSim.append(maskShift1)
            allFeatShift.append((featuresShift1*maskShift1).unsqueeze(0))
        allSim = torch.cat(allSim,dim=1)
        allFeatShift = torch.cat(allFeatShift,dim=0)

        return allSim,allFeatShift

    def forward(self,features):

        if self.normFeat:
            features = features/torch.sqrt(torch.pow(features,2).sum(dim=1,keepdim=True))

        simMap = computeTotalSim(features,1,self.neigAvgSize,self.reprVec)
        simMapList = [simMap]
        featList = [features]

        for j in range(self.nbIter):

            allSim,allFeatShift = self.computeSimAndShiftFeat(features)

            if self.random_farthest and j < 10:
                allSimDil,allFeatShiftDil = self.computeSimAndShiftFeat(features,self.neighDilation)
                mask = self.distr.sample((1,allSim.size(1),allSim.size(2),allSim.size(3))).bool().to(features.device).squeeze(-1)
                allSim = mask*allSim +(~mask)*(1-allSimDil)
                mask = mask.permute(1,0,2,3).unsqueeze(2)
                allFeatShift = mask*allFeatShift+(~mask)*allFeatShiftDil

            if self.weightByNeigSim and self.softmax:
                allSim = torch.softmax(self.softmax_fact*allSim,dim=1)

            allPondFeatShift = []
            for i in range(len(self.directions)):
                sim = allSim[:,i:i+1]
                featuresShift1 = allFeatShift[i]
                allPondFeatShift.append((sim*featuresShift1).unsqueeze(1))
            newFeatures = torch.cat(allPondFeatShift,dim=1).sum(dim=1)

            simSum = torch.nn.functional.conv2d(allSim,self.sumKer.to(allSim.device))
            newFeatures /= simSum

            features = newFeatures

            if self.normFeat:
                features = features/torch.sqrt(torch.pow(features,2).sum(dim=1,keepdim=True))

            featList.append(features)
            simMap = computeTotalSim(features,1,self.neigAvgSize,self.reprVec)
            simMapList.append(simMap)

        simMapList = torch.cat(simMapList,dim=1).unsqueeze(1)

        simMapList = simMapList.reshape(simMapList.size(0)//self.groupNb,self.groupNb,self.nbIter+1,simMapList.size(3),simMapList.size(4))
        # N x NbGroup x nbIter x sqrt(nbPatch) x sqrt(nbPatch)
        simMapList = simMapList.mean(dim=1)
        # N x nbIter x sqrt(nbPatch) x sqrt(nbPatch)

        return simMapList,featList

def buildModule(cond,constr,cuda,multi_gpu,kwargs={}):

    if cond:
        module = constr(**kwargs)
        if cuda:
            module = module.cuda()
            if multi_gpu:
                module = torch.nn.DataParallel(module)
    else:
        module = None
    return module

class GrahamProb(torch.nn.Module):
    def __init__(self):
        super(GrahamProb, self).__init__()

    def forward(self,feat):
        return grahamProbMap(feat)

class GrahamLoss(torch.nn.Module):
    def __init__(self):
        super(GrahamLoss,self).__init__()
    def forward(self,feat,feat_decod):
        featSize = feat.size(2)*feat.size(3)
        featNb = feat.size(1)
        grah_mat_enc = graham(feat)
        grah_mat_dec = graham(feat_decod)
        return (torch.pow(grah_mat_dec-grah_mat_enc,2).sum(dim=-1).sum(dim=-1)/(4*featSize*featSize*featNb*featNb))

class ReconstLoss(torch.nn.Module):
    def __init__(self):
        super(ReconstLoss,self).__init__()
    def forward(self,data,reconst):
        data = torch.nn.functional.adaptive_avg_pool2d(data, (reconst.size(-2), reconst.size(-1)))
        return torch.pow(reconst - data, 2).mean(dim=-1).mean(dim=-1).mean(dim=-1)

def grahamProbMap(feat):
    featCorr = (feat.unsqueeze(2)*feat.unsqueeze(1)).reshape(feat.size(0),feat.size(1)*feat.size(1),feat.size(2),feat.size(3))
    avgpoolKer = (torch.ones((14,14))/(14*14)).unsqueeze(0).unsqueeze(0).expand(featCorr.size(1),featCorr.size(1),-1,-1).to(feat.device)
    featCorr = torch.nn.functional.conv2d(featCorr,avgpoolKer)
    return computeTotalSim(featCorr,1)

def normResizeSave(destination,name,tensorToSave,i,min=None,max=None,size=336,imgPerRow=None):

    if not type(tensorToSave) is list:
        tensorToSave = [tensorToSave]
        suff = False
    else:
        suff = True

    for j in range(len(tensorToSave)):
        min = tensorToSave[j].min() if min is None else min
        max = tensorToSave[j].max() if max is None else max
        tensorToSave[j] = (tensorToSave[j]-min)/(max-min)
        tensorToSave[j] = torch.nn.functional.interpolate(tensorToSave[j], size=(size))

        if imgPerRow is None:
            grid = torchvision.utils.make_grid(tensorToSave[j])
        else:
            grid = torchvision.utils.make_grid(tensorToSave[j],nrow=imgPerRow)

        if type(destination) is str:
            torchvision.utils.save_image(grid, os.path.join(destination,name+("_{}".format(j) if suff else name)+".png"))
        else:
            destination.add_image(name+"_{}".format(j) if suff else name, grid, i)




def graham(feat,gramOrder):

    feat = feat.reshape(feat.size(0),feat.size(1),feat.size(2)*feat.size(3))

    if gramOrder == 1:
        gram = feat.sum(dim=-1)
    elif gramOrder == 2:
        gram = (feat.unsqueeze(2)*feat.unsqueeze(1)).sum(dim=-1)
        gram = gram.reshape(gram.size(0),gram.size(1)*gram.size(2))
    elif gramOrder == 3:
        featA = feat.unsqueeze(2).unsqueeze(3)
        featB = feat.unsqueeze(1).unsqueeze(3)
        featC = feat.unsqueeze(1).unsqueeze(2)
        gram = (featA*featB*featC).sum(dim=-1)
        gram = gram.reshape(gram.size(0),gram.size(1)*gram.size(2)*gram.size(3))
    else:
        raise ValueError("Unkown gram order :",gramOrder)
    return gram

if __name__ == "__main__":
    main()
