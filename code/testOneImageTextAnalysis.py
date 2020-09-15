import torch
import args
import modelBuilder
import load_data
import trainVal
from args import ArgReader
from tensorboardX import SummaryWriter
import torchvision
import sys
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
from scipy import ndimage
from skimage.transform import resize
import scipy.ndimage.filters as filters
import torch.nn.functional as F
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import cuml
import umap

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.enabled = False

def addArgs(argreader):
    argreader.parser.add_argument('--patch_sim', type=str2bool,
                                  help="To run a small experiment with an untrained CNN",default=True)
    argreader.parser.add_argument('--patch_sim_resnet', type=str2bool,
                                  help="To use resnet for the patch similarity experiment",default=True)
    argreader.parser.add_argument('--patch_sim_usemodel', type=str2bool,
                                  help="Set this to False to compare directly pixels",default=True)

    argreader.parser.add_argument('--patch_sim_restype', type=str,
                                  help="The resnet to use",default="resnet18")

    argreader.parser.add_argument('--patch_sim_relu_on_simple', type=str2bool,
                                  help="To add relu when using the simple CNN",default=True)

    argreader.parser.add_argument('--patch_sim_pretr_res', type=str2bool,
                                  help="To have the patch sim resnet pretrained on Imagenet",default=False)
    argreader.parser.add_argument('--patch_size', type=int,
                                  help="The patch size for the patch similarity exp",default=5)
    argreader.parser.add_argument('--patch_stride', type=int,
                                  help="The patch stride for the patch similarity exp",default=5)

    argreader.parser.add_argument('--patch_sim_group_nb', type=int,
                                  help="To compute gram matrix by grouping features using a random partitions to reduce RAM load.",default=1)

    argreader.parser.add_argument('--patch_sim_out_path', type=str,
                                  help="The output path",default="../../embyovis/")

    argreader.parser.add_argument('--patch_sim_gram_order', type=int,
                                  help="If 2, computes correlation between feature maps. If 1, computes average of feature maps.",default=1)

    argreader.parser.add_argument('--data_batch_index', type=int,
                                  help="The index of the batch to process first",default=1)

    argreader.parser.add_argument('--patch_sim_neighsim_nb_iter', type=int,
                                  help="The number of times to apply neighbor similarity averaging ",default=3)
    argreader.parser.add_argument('--patch_sim_neighsim_softmax', type=str2bool,
                                  help="Whether or not to use softmax to weight neighbors",default=True)
    argreader.parser.add_argument('--patch_sim_neighsim_softmax_fact', type=int,
                                  help="Whether or not to use softmax to weight neighbors",default=10)

    argreader.parser.add_argument('--patch_sim_weight_by_neigsim',type=str2bool,
                                  help="To weight a neighbor by similarity",default=True)
    argreader.parser.add_argument('--patch_sim_update_rate_by_cossim', type=str2bool,
                                  help="To define the update rate of the feature using the cossim map.",default=False)
    argreader.parser.add_argument('--patch_sim_neighradius', type=int,
                                  help="The radius of the neighborhood",default=1)
    argreader.parser.add_argument('--patch_sim_neighdilation', type=int,
                                  help="The dilation of the neighborhood",default=1)

    argreader.parser.add_argument('--crop', type=str2bool,
                                  help="To preprocess images by randomly cropping them",default=False)
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

    argreader.parser.add_argument('--repr_vec', type=str2bool,
                                  help="To run the representative vectors experiment",default=False)

    argreader.parser.add_argument('--centered_feat', type=str2bool,
                              help="To center the features before cosine similarity",default=False)
    argreader.parser.add_argument('--unif_sample', type=str2bool,
                              help="To use uniform sampling to select reference pixels when computing representative vectors.",default=False)
    argreader.parser.add_argument('--only_sim_sample', type=str2bool,
                              help="To sample only based on similarity",default=False)

    argreader.parser.add_argument('--redundacy', type=str2bool,
                              help="To compute pixel redundacy",default=False)

    argreader.parser.add_argument('--umap_nei_nb', type=int,
                              help="UMAP Neighbors number parameter.",default=15)
    argreader.parser.add_argument('--umap_min_dist', type=float,
                              help="UMAP Minimum distance parameter.",default=0.1)
    argreader.parser.add_argument('--apply_umap', type=str2bool,
                              help="To apply UMAP before computing representative vectors.",default=True)

    return argreader

def main(argv=None):
    # Getting arguments from config file and command line
    # Building the arg reader
    argreader = ArgReader(argv)

    argreader = addArgs(argreader)

    argreader = trainVal.addInitArgs(argreader)
    argreader = trainVal.addOptimArgs(argreader)
    argreader = trainVal.addValArgs(argreader)
    argreader = trainVal.addLossTermArgs(argreader)

    argreader = modelBuilder.addArgs(argreader)
    argreader = load_data.addArgs(argreader)

    # Reading the comand line arg
    argreader.getRemainingArgs()

    args = argreader.args

    if args.patch_sim:

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

        #args.second_mod = "pointnet2"
        args.pn_topk = True
        args.pn_topk_euclinorm = False
        args.texture_encoding = True
        args.big_images = True
        args.batch_size = 32
        args.val_batch_size = 32
        args.multi_gpu = True
        args.patch_sim_group_nb = 1
        #args.data_batch_index = 1

        if args.crop:
            transf = None
        else:
            transf = "identity"

        trainLoader, _ = load_data.buildTrainLoader(args,transf=transf,shuffle=args.shuffle)
        #trainLoader, _ = load_data.buildTestLoader(args, "test",shuffle=args.shuffle)

        data,target =next(iter(trainLoader))
        if args.data_batch_index>0:
            for i in range(args.data_batch_index):
                data,target =next(iter(trainLoader))

        #data = data[:32]

        with torch.no_grad():

            data = data.cuda() if args.cuda else data

            kwargsNet = {"resnet":args.patch_sim_resnet,"resType":args.patch_sim_restype,"pretr":args.patch_sim_pretr_res,"nbGroup":args.patch_sim_group_nb,\
                        "reluOnSimple":args.patch_sim_relu_on_simple,"chan":args.resnet_chan,"gramOrder":args.patch_sim_gram_order,"useModel":args.patch_sim_usemodel,\
                        "inChan":3,"resnet_multilev":args.resnet_multilev}

            kwargs = {"neigAvgSize":args.patch_sim_neiref_neisimavgsize,"reprVec":args.patch_sim_neiref_repr_vectors}
            cosimMap = buildModule(True,CosimMap,args.cuda,args.multi_gpu,kwargs)

            patch = data.unfold(2, args.patch_size, args.patch_stride).unfold(3, args.patch_size, args.patch_stride).permute(0,2,3,1,4,5)
            origPatchSize = patch.size()
            patch = patch.reshape(patch.size(0)*patch.size(1)*patch.size(2),patch.size(3),patch.size(4),patch.size(5))

            kwargsNeiSim = {"cuda":args.cuda,"groupNb":args.patch_sim_group_nb,"nbIter":args.patch_sim_neighsim_nb_iter,\
                        "softmax":args.patch_sim_neighsim_softmax,"softmax_fact":args.patch_sim_neighsim_softmax_fact,\
                        "weightByNeigSim":args.patch_sim_weight_by_neigsim,"updateRateByCossim":args.patch_sim_update_rate_by_cossim,"neighRadius":args.patch_sim_neighradius,\
                        "neighDilation":args.patch_sim_neighdilation,"random_farthest":args.random_farthest,"random_farthest_prop":args.random_farthest_prop,\
                        "neigAvgSize":args.patch_sim_neiref_neisimavgsize,"reprVec":args.patch_sim_neiref_repr_vectors,"normFeat":args.patch_sim_neiref_norm_feat,\
                        "centered":args.centered_feat}

            neighSimMod = buildModule(True,NeighSim,args.cuda,args.multi_gpu,kwargsNeiSim)

            repreKwargs = {"nbVec":10,"unifSample":args.unif_sample,"onlySimSample":args.only_sim_sample,"umapNeiNb":args.umap_nei_nb,\
                            "umapMinDist":args.umap_min_dist}

            representativeVectorsMod = buildModule(True,RepresentativeVectors,args.cuda,args.multi_gpu,repreKwargs)
            computeTotalSimMod = buildModule(True,ComputeTotalSim,args.cuda,args.multi_gpu,{})

            print("Start !")
            start = time.time()

            feat,distMapAgreg,neighSim,refFeatList = textLimit(patch,data,cosimMap,neighSimMod,kwargsNet,args)

            #for umap in [True,False]:
            reprVecStart = time.time()
            reprVec,simListRepr,selectedPos = representativeVectorsMod(refFeatList[-1],applyUMAP=args.apply_umap)
            print("Time to compute representative vectors",time.time()-reprVecStart)
            #_,simListReprNoUMAP,_ = representativeVectorsMod(refFeatList[-1],applyUMAP=False)
            #simListRepr_flat = simListReprNoUMAP.view(simListReprNoUMAP.size(0)*simListReprNoUMAP.size(1),1,simListReprNoUMAP.size(2),simListReprNoUMAP.size(3))
            #neighSim_redCosSim = computeTotalSim(simListRepr_flat,1,eucli=True)
            #neighSim_redCosSim = neighSim_redCosSim.view(simListReprNoUMAP.size(0),simListReprNoUMAP.size(1),neighSim_redCosSim.size(2),neighSim_redCosSim.size(3))
            #neighSim_redCosSimNoUMAP = neighSim_redCosSim.mean(dim=1,keepdim=True)

            if args.redundacy:
                simListReprAgr = (simListRepr*simListRepr.sum(dim=(2,3),keepdim=True)).sum(dim=1,keepdim=True)
                #simListReprAgr = (simListRepr*simListRepr.sum(dim=(2,3),keepdim=True)).max(dim=1,keepdim=True)[0]

                simListReprAgr_max = simListReprAgr.max(dim=-1,keepdim=True)[0].max(dim=-2,keepdim=True)[0]
                simListReprAgr_min = simListReprAgr.min(dim=-1,keepdim=True)[0].min(dim=-2,keepdim=True)[0]
                simListReprAgr_norm = (simListReprAgr-simListReprAgr_min)/(simListReprAgr_max-simListReprAgr_min)

                neighSim_red = torch.pow(neighSim[:,-1].unsqueeze(1),1/(simListReprAgr_norm+1))

                #print(simListReprAgr.size())
                simListRepr_flat = simListRepr.view(simListRepr.size(0)*simListRepr.size(1),1,simListRepr.size(2),simListRepr.size(3))
                neighSim_redCosSim = computeTotalSim(simListRepr_flat,1,eucli=True)
                neighSim_redCosSim = neighSim_redCosSim.view(simListRepr.size(0),simListRepr.size(1),neighSim_redCosSim.size(2),neighSim_redCosSim.size(3))
                neighSim_redCosSim = neighSim_redCosSim.mean(dim=1,keepdim=True)
                #neighSim_redCosSim = torch.pow(neighSim_redCosSim,10)

                coverageList = []
                for i in range(simListRepr.size(1)):
                    #coverageList.append(torch.abs(simListRepr[:,:i+1]).sum(dim=1,keepdim=True).unsqueeze(0)/simListRepr.size(1))
                    coverageList.append(simListRepr[:,:i+1].sum(dim=1,keepdim=True).unsqueeze(0))
                coverageList = torch.cat(coverageList,dim=0)
                coverageList_min = coverageList.min(dim=-1,keepdim=True)[0].min(dim=-2,keepdim=True)[0].min(dim=0,keepdim=True)[0]
                coverageList_max = coverageList.max(dim=-1,keepdim=True)[0].max(dim=-2,keepdim=True)[0].max(dim=0,keepdim=True)[0]
                coverageList = 255*(coverageList-coverageList_min)/(coverageList_max-coverageList_min)

                reprVec_exp = reprVec.unsqueeze(-1).unsqueeze(-1).expand(-1,-1,-1,simListRepr.size(-2),simListRepr.size(-1)).permute(0,1,3,4,2)
                # reprVec_exp.size() = N x 10 x 100 x 100 x C

                simListRepr_exp = torch.softmax(simListRepr.unsqueeze(-1),dim=1)
                #simListRepr_exp.size() = N x 10 x 100 x 100 x 1

                redRefinedFeat = (reprVec_exp*simListRepr_exp).sum(dim=1).permute(0,3,1,2)

                neighSim_redRefCosSim = computeTotalSim(redRefinedFeat,1,eucli=False)
                neighSim_redRefEucli = computeTotalSim(redRefinedFeat,1,eucli=True)

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

            neighSim = neighSim.detach().cpu().numpy()

            simMapPath = os.path.join(args.patch_sim_out_path,"simMaps",args.model_id,str(i+args.data_batch_index*args.batch_size))

            for i,img in enumerate(data):

                #if i < 9:
                if not os.path.exists(os.path.join(args.patch_sim_out_path,"simMaps",args.model_id,"{}".format(i+args.data_batch_index*args.batch_size))):
                    os.makedirs(os.path.join(args.patch_sim_out_path,"simMaps",args.model_id,"{}".format(i+args.data_batch_index*args.batch_size)))

                #img = (img-img.min())/(img.max()-img.min())

                #plotImg(img.detach().cpu().permute(1,2,0).numpy(),os.path.join(args.patch_sim_out_path,"imgs","{}.png".format(i+args.data_batch_index*args.batch_size)))

                #resizedImg = resize(img.detach().cpu().permute(1,2,0).numpy().mean(axis=-1), (100,100),anti_aliasing=True,mode="constant",order=0)*255
                #edges = ~skimage.feature.canny(img.detach().cpu().permute(1,2,0).numpy().mean(axis=-1),sigma=1)
                #plotImg(edges,os.path.join(args.patch_sim_out_path,"edges","sig1_{}.png".format(i+args.data_batch_index*args.batch_size)))
                #edges = ~skimage.feature.canny(img.detach().cpu().permute(1,2,0).numpy().mean(axis=-1),sigma=3)
                #plotImg(edges,os.path.join(args.patch_sim_out_path,"edges","sig3_{}.png".format(i+args.data_batch_index*args.batch_size)))
                #edges = ~skimage.feature.canny(img.detach().cpu().permute(1,2,0).numpy().mean(axis=-1),sigma=9)
                #plotImg(edges,os.path.join(args.patch_sim_out_path,"edges","sig9_{}.png".format(i+args.data_batch_index*args.batch_size)))

                #sobel = sobelFunc(img.detach().cpu().permute(1,2,0).numpy().mean(axis=-1))
                #plotImg(sobel,os.path.join(args.patch_sim_out_path,"sobel","{}.png".format(i+args.data_batch_index*args.batch_size)))

                #sobel = sobelFunc(img.detach().cpu().permute(1,2,0).numpy().mean(axis=-1))
                #minima,_,_ = computeMinima(sobel)
                #plotImg(255-((255-sobel)*minima),os.path.join(args.patch_sim_out_path,"sobel","nms-{}.png".format(i+args.data_batch_index*args.batch_size)))
                #topk(sobel,minima,os.path.join(args.patch_sim_out_path,"sobel"),"{}".format(i+args.data_batch_index*args.batch_size))

                simMapPath = os.path.join(args.patch_sim_out_path,"simMaps",args.model_id,str(i+args.data_batch_index*args.batch_size))

                writeAllImg(distMapAgreg,neighSim,i,simMapPath,"sparseNeighSim_step0")

                #dimRedList(refFeatList,simMapPath,i,"neiRef")
                #plotNorm(refFeatList,i,simMapPath)
                #if not refFeatRepList is None:
                #    dimRedList(refFeatRepList,simMapPath,i,"neiRef_repr")
                #    for j in range(len(neighSimRepList)):
                #        pathPNG = os.path.join(simMapPath,"sparseNeighSim_step{}_repr".format(len(neighSimRepList)-1-j))
                #        plotImg(neighSimRepList[j][i][0],pathPNG,cmap="gray")

                writeReprVecSimMaps(simListRepr,i,simMapPath)

                if args.redundacy:
                    writeRed(simListReprAgr,neighSim_red,neighSim_redCosSim,coverageList,neighSim_redRefCosSim,neighSim_redRefEucli,i,simMapPath)
                    normAndPlot(selectedPos[i,0],os.path.join(simMapPath,"selPos.png"))

                #normAndPlot(neighSim_redCosSimNoUMAP[i,0],os.path.join(simMapPath,"redun_neighSimCosSimNoUMAP.png"))

                #writeRed(neighSim_redCosSim,i,simMapPath)


def dimRedBatch(data,simMapPath):

    print(data.size())

    origSize = data.size()
    data = data.permute(0,2,3,1).reshape(-1,data.size(1)).cpu().detach().numpy()

    # (X 3)
    print(data.shape)
    emb = umap.UMAP(n_components=3).fit_transform(data)

    # (N H W 3)
    emb = emb.reshape((origSize[0],origSize[2],origSize[3],3))
    emb = (255*(emb-emb.min())/(emb.max()-emb.min())).astype("uint8")

    for i in range(len(emb)):
        plotImg(emb[i],os.path.join(simMapPath,"umapBatch_{}".format(i)))

def normAndPlot(img,simMapPath,norm=True):
    if norm:
        img = (255*(img-img.min())/(img.max()-img.min()))
    img = img.cpu().numpy()
    plotImg(img,simMapPath)

def writeRed(simListReprAgr,neighSim_red,neighSim_redCosSim,coverageList,neighSim_redRefCosSim,neighSim_redRefEucli,i,simMapPath):
    normAndPlot(simListReprAgr[i,0],os.path.join(simMapPath,"redun.png"))
    normAndPlot(neighSim_red[i,0],os.path.join(simMapPath,"redun_neighSim.png"))
    normAndPlot(neighSim_redCosSim[i,0],os.path.join(simMapPath,"redun_neighSimCosSim.png"))
    for j,coverage in enumerate(coverageList):
        #print(coverage[i,0].min().item(),coverage[i,0].float().mean().item(),coverage[i,0].max().item())
        normAndPlot(coverage[i,0],os.path.join(simMapPath,"redun_coverage{}.png".format(j)),norm=False)
    normAndPlot(neighSim_redRefCosSim[i,0],os.path.join(simMapPath,"redun_refNeighSimCosSim.png"))
    normAndPlot(neighSim_redRefEucli[i,0],os.path.join(simMapPath,"redun_refNeighSimEucli.png"))

def plotNorm(refFeatList,i,simMapPath):

    for j in range(len(refFeatList)):
        norm = torch.sqrt(torch.pow(refFeatList[j][i:i+1],2).sum(dim=1))[0].detach().cpu().numpy()
        plotImg(norm,os.path.join(simMapPath,"norm_{}.png".format(j)),cmap="gray")

def writeReprVecSimMaps(simListRepr,i,simMapPath):
    for j in range(len(simListRepr[i])):
        path = os.path.join(simMapPath,"reprVecSimMaps_v{}.png".format(j))

        img_np = simListRepr[i,j].cpu().detach().numpy()
        img_np = 255*(img_np-img_np.min())/(img_np.max()-img_np.min())

        plotImg(img_np,path)
        #cv2.imwrite(path,img_np)

def textLimit(patch,data,cosimMap,neighSimMod,kwargsNet,args):
    #neighSimList = []
    #for i in range(20):
    kwargsNet["inChan"] = patch.size(1)
    patchMod = buildModule(True,PatchSimCNN,args.cuda,args.multi_gpu,kwargsNet)

    featStart = time.time()

    if kwargsNet["resType"].find("bagnet") == -1:
        gram_mat = patchMod(patch)
        gram_mat = gram_mat.view(data.size(0),-1,args.patch_sim_group_nb,gram_mat.size(2))
        gram_mat = gram_mat.permute(0,2,1,3)
        distMapAgreg,feat = cosimMap(gram_mat)
        print("Feat size",feat.size())
    else:
        feat = patchMod(data)
        distMapAgreg = computeTotalSim(feat,1)

    print("Time to compute features",time.time()-featStart)

    refFeatStart = time.time()
    neighSim,refFeatList = neighSimMod(feat)
    print("Time to refine features",time.time()-refFeatStart)

    return feat,distMapAgreg,neighSim,refFeatList

def writeAllImg(distMapAgreg,neighSim,i,simMapPath,name):
    if not distMapAgreg is None:
        plotImg(distMapAgreg[i][0][1:-1,1:-1].detach().cpu().numpy(),os.path.join(simMapPath,name+"_agr.png"),'gray')

    neighSim[i] = 255*(neighSim[i]-neighSim[i].min())/(neighSim[i].max()-neighSim[i].min())
    for j in range(len(neighSim[i])):
        pathPNG = os.path.join(simMapPath,name.replace("step0","step{}".format(len(neighSim[i])-1-j)))
        plotImg(neighSim[i][j],pathPNG,cmap="gray")

def dimRed(refFeat,simMapPath,i,name):

    refFeat_emb = umap.UMAP(n_components=3).fit_transform(refFeat[i].view(refFeat[i].size(0),-1).permute(1,0).cpu().detach().numpy())
    refFeat_emb = refFeat_emb.reshape((refFeat[i].size(1),refFeat[i].size(2),3))
    refFeat_emb = (refFeat_emb-refFeat_emb.min())/(refFeat_emb.max()-refFeat_emb.min())

    plt.figure(figsize=(10,10))
    plt.imshow(refFeat_emb)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(os.path.join(simMapPath,"{}_tsne_img.png".format(name)))
    plt.close()

    refFeat_emb_tens = torch.tensor(refFeat_emb).to(refFeat.device).permute(2,0,1).unsqueeze(0)
    neighSim = computeTotalSim(refFeat_emb_tens,1,1,False)[0,0].detach().cpu().numpy()
    plt.figure(figsize=(10,10))
    plt.imshow(neighSim)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.savefig(os.path.join(simMapPath,"{}_tsne_img_cosim.png".format(name)))
    plt.close()

def dimRedList(refFeatList,simMapPath,i,name):

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

        #refFeat_emb_tens = torch.tensor(refFeat_emb[j]).to(refFeatList[0].device).permute(2,0,1).unsqueeze(0)
        #neighSim = computeTotalSim(refFeat_emb_tens,1,1,False)[0,0].detach().cpu().numpy()
        #plt.figure(figsize=(10,10))
        #plt.imshow(neighSim,cmap="gray")
        #plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        #plt.savefig(os.path.join(simMapPath,"{}_tsne_img_cosim.png".format(name)))
        #plt.close()

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
                if resType.find("bagnet") == -1:
                    kwargs["chan"] = kwargs["chan"]//2
                    self.featMod = modelBuilder.buildFeatModel(resType, pretr, True, False,**kwargs)
                else:
                    kwargs.pop('chan', None)
                    kwargs.pop('inChan', None)
                    self.featMod = modelBuilder.buildFeatModel(resType, pretr, True, False,**kwargs)

        self.resType = resType
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

        if self.resType.find("bagnet") == -1:
            origFeatSize = featVolume.size()
            featVolume = featVolume.unfold(1, featVolume.size(1)//self.nbGroup, featVolume.size(1)//self.nbGroup).permute(0,1,4,2,3)
            featVolume = featVolume.view(featVolume.size(0)*featVolume.size(1),featVolume.size(2),featVolume.size(3),featVolume.size(4))
            gramMat = graham(featVolume,self.gramOrder)
            #gramMat = gramMat.view(origFeatSize[0],self.nbGroup,origFeatSize[1]//self.nbGroup,origFeatSize[1]//self.nbGroup)
            gramMat = gramMat.view(origFeatSize[0],self.nbGroup,gramMat.size(-1))
            return gramMat
        else:
            return featVolume

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

    maskShift = maskShift.mean(dim=1,keepdim=True)
    return featuresShift,maskShift

def applyDiffKer_CosSimi(direction,features,dilation=1,neigAvgSize=1,reprVec=False,centered=False,eucli=False):
    origFeatSize = features.size()
    featNb = origFeatSize[1]

    if centered:
        features = features - features.mean(dim=-1,keepdim=True).mean(dim=-2,keepdim=True)

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

    if eucli:
        sim = torch.exp(-torch.sqrt(torch.pow(featuresShift1-featuresShift2,2).sum(dim=1,keepdim=True)*maskShift1*maskShift2)/20)
        #sim = torch.exp(-torch.sqrt(torch.pow(x-raw_reprVec,2).sum(dim=-1))/20)
    else:
        sim = (featuresShift1*featuresShift2*maskShift1*maskShift2).sum(dim=1,keepdim=True)
        sim /= torch.sqrt(torch.pow(maskShift1*featuresShift1,2).sum(dim=1,keepdim=True))*torch.sqrt(torch.pow(maskShift2*featuresShift2,2).sum(dim=1,keepdim=True))

    return sim,featuresShift1,featuresShift2,maskShift1,maskShift2

def computeTotalSim(features,dilation,neigAvgSize=1,reprVec=False,centered=False,eucli=False):
    horizDiff,_,_,_,_ = applyDiffKer_CosSimi("horizontal",features,dilation,neigAvgSize,reprVec,centered,eucli)
    vertiDiff,_,_,_,_ = applyDiffKer_CosSimi("vertical",features,dilation,neigAvgSize,reprVec,centered,eucli)
    totalDiff = (horizDiff + vertiDiff)/2
    return totalDiff

class NeighSim(torch.nn.Module):
    def __init__(self,cuda,groupNb,nbIter,softmax,softmax_fact,weightByNeigSim,updateRateByCossim,neighRadius,neighDilation,random_farthest,random_farthest_prop,\
                        neigAvgSize,reprVec,normFeat,centered=False):
        super(NeighSim,self).__init__()

        self.directions = computeNeighborsCoord(neighRadius)

        self.sumKer = torch.ones((1,len(self.directions),1,1))
        self.sumKer = self.sumKer.cuda() if cuda else self.sumKer
        self.groupNb = groupNb
        self.nbIter = nbIter
        self.softmax = softmax
        self.softmax_fact = softmax_fact
        self.centered = centered

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
                sim,featuresShift1,_,maskShift1,_ = applyDiffKer_CosSimi(direction*dilation,features,1,self.neigAvgSize,self.reprVec,self.centered)
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

        simMap = computeTotalSim(features,1,self.neigAvgSize,self.reprVec,self.centered,eucli=True)
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
            simMap = computeTotalSim(features,1,self.neigAvgSize,self.reprVec,self.centered,eucli=True)
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

class RepresentativeVectors(torch.nn.Module):
    def __init__(self,nbVec,unifSample,onlySimSample,umapNeiNb,umapMinDist):
        super(RepresentativeVectors,self).__init__()
        self.nbVec = nbVec
        self.unifSample = unifSample
        self.onlySimSample = onlySimSample
        self.umapNeiNb = umapNeiNb
        self.umapMinDist = umapMinDist

    def forward(self,x,applyUMAP=True):
        vecList,simList,selectedPos = representativeVectors(x,self.nbVec,unifSample=self.unifSample,onlySimSample=self.onlySimSample,applyUMAP=applyUMAP,\
                                                            umapNeiNb=self.umapNeiNb,umapMinDist=self.umapMinDist)

        vecList = list(map(lambda x:x.unsqueeze(1),vecList))
        vecList = torch.cat(vecList,dim=1)
        simList = torch.cat(simList,dim=1)
        return vecList,simList,selectedPos

def representativeVectors(x,nbVec,prior=None,unifSample=False,onlySimSample=False,applyUMAP=True,umapNeiNb=15,umapMinDist=0.1):

    if applyUMAP:
        emb_list = []
        for i in range(len(x)):
            #emb = PCA(n_components=2,random_state=1).fit_transform(x[i].reshape(x[i].size(0),-1).permute(1,0).cpu().detach().numpy())
            #emb = umap.UMAP(n_components=2,random_state=1,n_neighbors=umapNeiNb,min_dist=umapMinDist).fit_transform(x[i].reshape(x[i].size(0),-1).permute(1,0).cpu().detach().numpy())
            emb = cuml.UMAP(n_components=2,n_neighbors=umapNeiNb,min_dist=umapMinDist).fit_transform(x[i].reshape(x[i].size(0),-1).permute(1,0).contiguous().cpu().detach().numpy())
            emb = torch.tensor(emb).to(x.device).reshape(x[i].size(-2),x[i].size(-1),-1).permute(2,0,1).unsqueeze(0)
            emb_list.append(emb)
        x = torch.cat(emb_list,dim=0)

    xOrigShape = x.size()
    normNotFlat = torch.sqrt(torch.pow(x,2).sum(dim=1,keepdim=True))
    x = x.permute(0,2,3,1).reshape(x.size(0),x.size(2)*x.size(3),x.size(1))
    norm = torch.sqrt(torch.pow(x,2).sum(dim=-1)) + 0.00001

    selectedPos = torch.zeros(xOrigShape[0],xOrigShape[2]*xOrigShape[3]).to(x.device)

    if prior is None:
        raw_reprVec_score = torch.rand(norm.size()).to(norm.device)
        #raw_reprVec_score = torch.ones_like(norm).to(norm.device)
        #raw_reprVec_score[:,norm.size(-1)//2] = 2
    else:
        prior = prior.reshape(prior.size(0),-1)
        raw_reprVec_score = prior.clone()

    repreVecList = []
    simList = []
    for i in range(nbVec):

        if unifSample:
            ind = i*(x.size(1)//nbVec)
            raw_reprVec_norm = norm[:,ind].unsqueeze(-1)
        elif onlySimSample:
            if i == 0:
                ind = x.size(1)//2
                raw_reprVec_norm = norm[:,ind].unsqueeze(-1)
                selectedPos[:,ind] += 1
            else:
                _,ind = raw_reprVec_score.max(dim=1,keepdim=True)
                raw_reprVec_norm = norm[torch.arange(x.size(0)).unsqueeze(1),ind]
                selectedPos[torch.arange(x.size(0)).unsqueeze(1),ind] += 1
        else:
            _,ind = raw_reprVec_score.max(dim=1,keepdim=True)
            raw_reprVec_norm = norm[torch.arange(x.size(0)).unsqueeze(1),ind]

        raw_reprVec = x[torch.arange(x.size(0)).unsqueeze(1),ind]
        #sim = (x*raw_reprVec).sum(dim=-1)/(norm*raw_reprVec_norm)
        #simNorm = torch.softmax(2*sim,dim=1)

        #dist = torch.sqrt(torch.pow(x-raw_reprVec,2).sum(dim=-1))
        #sim = (dist.max(dim=-1,keepdim=True)[0]-dist)
        #sim = (x*raw_reprVec).sum(dim=-1)/(norm*raw_reprVec_norm)

        #for img in torch.sqrt(torch.pow(x-raw_reprVec,2).sum(dim=-1)):
        #    plt.figure()
        #    plt.hist(img.cpu().numpy())
        #    plt.savefig("../vis/distDistri.png")
        #    plt.close()
        #sys.exit(0)

        sim = torch.exp(-torch.sqrt(torch.pow(x-raw_reprVec,2).sum(dim=-1))/20)

        #sim_min = sim.min(dim=1,keepdim=True)[0]
        #sim_max = sim.max(dim=1,keepdim=True)[0]
        #sim = torch.clamp((sim - sim_min)/(sim_max - sim_min),0.0001,1)
        simNorm = sim/sim.sum(dim=1,keepdim=True)

        reprVec = (x*simNorm.unsqueeze(-1)).sum(dim=1)
        repreVecList.append(reprVec)

        if onlySimSample and i == 0:
            raw_reprVec_score = (1-sim)
        else:
            raw_reprVec_score = (1-sim)*raw_reprVec_score

        simReshaped = sim.reshape(sim.size(0),1,xOrigShape[2],xOrigShape[3])

        simList.append(simReshaped)

    selectedPos = selectedPos.reshape(selectedPos.size(0),1,xOrigShape[2],xOrigShape[3])

    return repreVecList,simList,selectedPos

if __name__ == "__main__":
    main()
