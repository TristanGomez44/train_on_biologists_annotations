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
def main(argv=None):
    # Getting arguments from config file and command line
    # Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--layer', type=int,
                                  help="The encoding layer to use for feature similarity.",default=4)

    argreader.parser.add_argument('--gram_matrix_weight', type=float,
                                  help="The weight of the gram matrix loss term",default=0)
    argreader.parser.add_argument('--gram_matrix_weight_threshold', type=float,
                                  help="The threshold at which the graham term is applied. If the term weight is positive, the term stops being applied \
                                  it gets below the threshold. If the weight is negative, the term stops being applied if it gets above.",default=0)
    argreader.parser.add_argument('--simclr_weight', type=float,
                                  help="The weight of the SimCLR term",default=0)
    argreader.parser.add_argument('--simclr_projdim', type=int,
                                  help="For simCLR : The projection dimension",default=64)

    argreader.parser.add_argument('--separated_training', type=str2bool,
                                  help="Trains an encoder to maximise reconstruction and trains a decoder to minimise graham matrix difference \
                                  between encoding and decoding",default=False)

    argreader.parser.add_argument('--last_conv_decod', type=str2bool,
                                  help="Compute the prob map on the last conv layer of the decoder",default=False)

    argreader.parser.add_argument('--lay_1_conv_decod', type=str2bool,
                                  help="Compute the prob map on the feature of layer 1 of the decoder",default=False)

    argreader.parser.add_argument('--patch_sim', type=str2bool,
                                  help="To run a small experiment with an untrained CNN",default=False)
    argreader.parser.add_argument('--patch_sim_resnet', type=str2bool,
                                  help="To use resnet for the patch similarity experiment",default=False)
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

    argreader.parser.add_argument('--pixel_sim', type=str2bool,
                                  help="To run a small experiment with an untrained CNN",default=False)
    argreader.parser.add_argument('--pixel_sim_kersize', type=int,
                                  help="kernel size for pixel similarity",default=10)
    argreader.parser.add_argument('--pixel_sim_stride', type=int,
                                  help="stride for pixel similarity",default=4)
    argreader.parser.add_argument('--patch_sim_kertype', type=str,
                                  help="Kernel type. Can be linear, squarred or constant.",default="linear")

    argreader.parser.add_argument('--data_batch_index', type=int,
                                  help="The index of the batch to process first",default=0)


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

    if not args.separated_training:
        optim = getattr(torch.optim,args.optim)(net.parameters(),lr=args.lr, **kwargs)
    else:
        #encParams = list(net.firstModel.featMod.parameters())+list(net.firstModel.bottleneck.parameters())
        optimEnc = getattr(torch.optim,args.optim)(net.parameters(),lr=args.lr, **kwargs)
        decParams = list(net.firstModel.decoder.parameters())
        optimDec = getattr(torch.optim,args.optim)(decParams,lr=args.lr, **kwargs)

    if args.simclr_weight > 0:
        mask = mask_correlated_samples(args)
        simCLR_loss = nt_xent.NT_Xent(args.batch_size, 0.5, mask)
        simclrDict = {"weight":args.simclr_weight,"loss":simCLR_loss}
        transf = transfSimCLR.TransformsSimCLR()
        nbFeat = args.pn_enc_chan
        simclrDict["mlp"] = torch.nn.Sequential(torch.nn.Linear(nbFeat, nbFeat, bias=False),torch.nn.ReLU(),torch.nn.Linear(nbFeat, args.simclr_projdim, bias=False))
        simclrDict["mlp"] = simclrDict["mlp"].cuda() if args.cuda else simclrDict["mlp"]
    else:
        simCLR_loss = None
        simclrDict = None
        transf = None

    trainLoader, _ = load_data.buildTrainLoader(args,transf=transf if args.simclr_weight>0 else None,shuffle=args.simclr_weight==0)

    data,target =next(iter(trainLoader))
    if args.data_batch_index>0:
        for i in range(args.data_batch_index):
            data,target =next(iter(trainLoader))

    gram_loss_module = buildModule(args.gram_matrix_weight > 0,GrahamLoss,args.cuda,args.multi_gpu)
    reconst_loss_module = buildModule(args.reconst_weight > 0,ReconstLoss,args.cuda,args.multi_gpu)
    grahProbModule = GrahamProb()
    if args.multi_gpu:
        grahProbModule = torch.nn.DataParallel(grahProbModule)

    writer = SummaryWriter('../results/{}'.format(args.exp_id))

    normResizeSave(writer,'{}_in_images'.format(args.model_id),data,0)

    if (not args.patch_sim) and (not args.pixel_sim):

        for i in range(args.epochs):
            if i%args.log_interval == 0:
                print(i)

            #data_noise = data+torch.zeros_like(data).normal_(0, data.std()/2)
            #normResizeSave(writer,'{}_in_images_noise'.format(args.model_id),data_noise,i)

            if args.simclr_weight>0:
                #data1,data2 =transf(data)
                data,target = next(iter(trainLoader))
                normResizeSave(writer,'{}_in_images'.format(args.model_id),data,i)
                data1,data2 = data
                if args.cuda:
                    data1,data2 = data1.cuda(),data2.cuda()
                retDict1,retDict2 = net(data1),net(data2)
                h1,h2 = retDict1["features"].mean(dim=-1).mean(dim=-1),retDict2["features"].mean(dim=-1).mean(dim=-1)
                simclrDict["z1"],simclrDict["z2"] = simclrDict["mlp"](h1),simclrDict["mlp"](h2)
                retDict = retDict1
            else:
                if args.cuda:
                    data = data.cuda()
                retDict = net(data)

            lossDict,prob_map,reconst = computeLoss(args.reconst_weight,reconst_loss_module,args.text_enc_weight, args.neigh_pred_weight,args.gram_matrix_weight,\
                                                                        args.gram_matrix_weight_threshold,gram_loss_module,simclrDict,\
                                                                        retDict,data,target,args.text_enc_pos_dil,args.text_enc_neg_dil,\
                                                                        args.text_env_margin,args.layer)

            total_loss,loss,graham_loss,simCLR_loss = lossDict["total_loss"],lossDict["loss"],lossDict["graham_loss"],lossDict["simCLR_loss"]

            if not args.separated_training:
                total_loss.backward()
                optim.step()
                optim.zero_grad()
            else:
                loss.backward(retain_graph=True)
                optimEnc.step()
                optimEnc.zero_grad()
                graham_loss.backward()
                optimDec.step()
                optimDec.zero_grad()

            writer.add_scalars("Loss",{args.model_id+"_total":total_loss.cpu().detach()},i)
            writer.add_scalars("Loss",{args.model_id:loss.cpu().detach()},i)
            if not type(graham_loss) is int:
                writer.add_scalars("Loss",{args.model_id+"_graham":graham_loss.cpu().detach()},i)

            if i%10 == 0:
                normResizeSave(writer,'{}_prob_map'.format(args.model_id),prob_map,i)

                if "neighFeatPredErr" in retDict.keys():
                    normResizeSave(writer,'{}_neigh_pred_error'.format(args.model_id),-retDict["neighFeatPredErr"],i)

                if not reconst is None:
                    normResizeSave(writer,'{}_reconst'.format(args.model_id),reconst,i)

                if args.last_conv_decod:
                    with torch.no_grad():
                        prob_map_last_conv = modelBuilder.computeTotalSim(retDict["layerFeat_decod"]["conv1"],args.text_enc_pos_dil)
                        normResizeSave(writer,'{}_prob_map_lastconv'.format(args.model_id),prob_map_last_conv,i)

                if args.lay_1_conv_decod:
                    with torch.no_grad():
                        prob_map_last_lay1 = modelBuilder.computeTotalSim(retDict["layerFeat_decod"][1],args.text_enc_pos_dil)
                        normResizeSave(writer,'{}_prob_map_declay1'.format(args.model_id),prob_map_last_lay1,i)

    elif args.patch_sim:
        with torch.no_grad():
            print("Start !")
            start = time.time()

            kwargs = {"full_mode":args.patch_sim_full,"kerSize":args.patch_sim_neig_mode_ker_size,"parralelMode":args.patch_sim_paral_mode}
            gramDistMap = buildModule(True,GramDistMap,args.cuda,args.multi_gpu,kwargs)

            kwargs = {"resnet":args.patch_sim_resnet,"resType":args.patch_sim_restype,"pretr":args.patch_sim_pretr_res,"nbGroup":args.patch_sim_group_nb,\
                        "reluOnSimple":args.patch_sim_relu_on_simple,"chan":args.resnet_chan}
            net = buildModule(True,PatchSimCNN,args.cuda,args.multi_gpu,kwargs)

            patch = data.unfold(2, args.patch_size, args.patch_stride).unfold(3, args.patch_size, args.patch_stride).permute(0,2,3,1,4,5)

            origPatchSize = patch.size()
            patch = patch.reshape(patch.size(0)*patch.size(1)*patch.size(2),patch.size(3),patch.size(4),patch.size(5))

            gram_mat = net(patch)
            gram_mat = gram_mat.view(data.size(0),-1,args.patch_sim_group_nb,gram_mat.size(2),gram_mat.size(3))
            gram_mat = gram_mat.permute(0,2,1,3,4)

            distMap = gramDistMap(gram_mat,patchPerRow=origPatchSize[2])

            print("End ",time.time()-start)

            if not os.path.exists("../vis/{}/".format(args.exp_id)):
                os.makedirs("../vis/{}/".format(args.exp_id))
            if not os.path.exists("../results/{}/".format(args.exp_id)):
                os.makedirs("../results/{}/".format(args.exp_id))

            patch = patch.reshape(origPatchSize).detach().cpu().numpy()

            if not os.path.exists(os.path.join(args.patch_sim_out_path,"imgs")):
                os.makedirs(os.path.join(args.patch_sim_out_path,"imgs"))
            if not os.path.exists(os.path.join(args.patch_sim_out_path,"simMaps")):
                os.makedirs(os.path.join(args.patch_sim_out_path,"simMaps"))
            if not os.path.exists(os.path.join(args.patch_sim_out_path,"simMaps",args.model_id)):
                os.makedirs(os.path.join(args.patch_sim_out_path,"simMaps",args.model_id))

            if args.patch_size == args.patch_stride:
                distMap = distMap[:,:,:,np.newaxis].reshape(distMap.shape[0],distMap.shape[1],origPatchSize[1],origPatchSize[2])
            distMap = distMap.detach()
            distMap = torch.nn.functional.interpolate(distMap, size=(data.size(-1),data.size(-2)))
            distMap = distMap.cpu().numpy()

            for i,img in enumerate(data):

                if not os.path.exists(os.path.join(args.patch_sim_out_path,"simMaps",args.model_id,"{}".format(i+args.data_batch_index*args.batch_size))):
                    os.makedirs(os.path.join(args.patch_sim_out_path,"simMaps",args.model_id,"{}".format(i+args.data_batch_index*args.batch_size)))

                img = (img-img.min())/(img.max()-img.min())

                plt.figure(figsize=(10,10))
                plt.imshow(img.detach().cpu().permute(1,2,0).numpy())
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                plt.savefig(os.path.join(args.patch_sim_out_path,"imgs","{}.png".format(i+args.data_batch_index*args.batch_size)))
                plt.close()

                mask = (distMap[i] != -1)
                distMap[i][mask] = (distMap[i][mask]-distMap[i][mask].min())/(distMap[i][mask].max()-distMap[i][mask].min())

                #with imageio.get_writer("../vis/{}/points_{}_distmap_{}.gif".format(args.exp_id,args.model_id,i), mode='I',fps=1) as writer:

                for l in range(origPatchSize[1]):
                    for m in range(origPatchSize[2]):
                        refPatchInd = l*origPatchSize[2]+m
                        if args.patch_size == args.patch_stride:
                            map = distMap[i][refPatchInd]
                        else:
                            plt.imshow(img.detach().cpu().permute(1,2,0).numpy())
                            for j in range(origPatchSize[1]):
                                for k in range(origPatchSize[2]):
                                    if j !=l or k != m:

                                        patchInd = j*origPatchSize[2]+k

                                        x = int(k*data.size(2)/origPatchSize[1])
                                        y = int(j*data.size(3)/origPatchSize[2])
                                        alpha = distMap[i][refPatchInd,patchInd]

                                        rect = patches.Rectangle((x,y), args.patch_size, args.patch_size,alpha=(alpha!=-1),linewidth=0,color=cm.plasma(alpha))
                                        ax.add_patch(rect)

                        #refX = int(m*data.size(2)/origPatchSize[1])
                        #refY = int(l*data.size(3)/origPatchSize[2])
                        #refRect = patches.Rectangle((refX,refY), args.patch_size, args.patch_size,alpha=1,linewidth=0,color="black")
                        #ax.add_patch(refRect)

                        #array = fig2data(fig)
                        #writer.append_data(img_as_ubyte(array.astype("uint8")))

                        #plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                        #plt.savefig(os.path.join(args.patch_sim_out_path,"simMaps",args.model_id,"{}".format(i),"{}_{}.png".format(m,l)))
                        #array = fig2data(fig)
                        cv2.imwrite(os.path.join(args.patch_sim_out_path,"simMaps",args.model_id,"{}".format(i+args.data_batch_index*args.batch_size),"{}_{}.png".format(m,l)),255*map)
                        #plt.close()

    else:
        with torch.no_grad():
            print("Start !")
            start = time.time()

            kwargs = {"resnet":args.patch_sim_resnet,"pretr":args.patch_sim_pretr_res,"kerSize":args.pixel_sim_kersize,"groupNb":args.patch_sim_group_nb,\
                        "stride":args.pixel_sim_stride,"kerType":args.patch_sim_kertype}
            net = buildModule(True,PxlSimCNN,args.cuda,args.multi_gpu,kwargs)
            data = data.cuda() if args.cuda else data
            simMap = net(data)
            print("End ",time.time()-start)
            #if not os.path.exists("../vis/{}/".format(args.exp_id)):
            #    os.makedirs("../vis/{}/".format(args.exp_id))
            #if not os.path.exists("../results/{}/".format(args.exp_id)):
            #    os.makedirs("../results/{}/".format(args.exp_id))

            #if not os.path.exists(os.path.join(args.patch_sim_out_path,"imgs")):
            #    os.makedirs(os.path.join(args.patch_sim_out_path,"imgs"))
            #if not os.path.exists(os.path.join(args.patch_sim_out_path,"simMaps")):
            #    os.makedirs(os.path.join(args.patch_sim_out_path,"simMaps"))
            #if not os.path.exists(os.path.join(args.patch_sim_out_path,"simMaps",args.model_id)):
            #    os.makedirs(os.path.join(args.patch_sim_out_path,"simMaps",args.model_id))

            for i,img in enumerate(data):

                #if not os.path.exists(os.path.join(args.patch_sim_out_path,"simMaps",args.model_id,"{}".format(i))):
                #    os.makedirs(os.path.join(args.patch_sim_out_path,"simMaps",args.model_id,"{}".format(i)))

                img = (img-img.min())/(img.max()-img.min())

                plt.figure(figsize=(10,10))
                #plt.imshow(img.detach().cpu().permute(1,2,0).numpy())
                #plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                #plt.savefig(os.path.join(args.patch_sim_out_path,"imgs","{}.png".format(i)))
                #plt.close()

                #mask = (distMap[i] != -1)
                #distMap[i][mask] = (distMap[i][mask]-distMap[i][mask].min())/(distMap[i][mask].max()-distMap[i][mask].min())

                simMap[i] = (simMap[i]-simMap[i].min())/(simMap[i].max()-simMap[i].min())

                plt.imshow(simMap[i][0][1:-1,1:-1].detach().cpu().numpy())
                plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
                plt.savefig("../vis/{}/points_{}_simMap_{}.png".format(args.exp_id,args.model_id,i))
                plt.close()


def fig2data(fig):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()

    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.frombuffer ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( h, w,4 )

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf


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
            gram = gram.reshape(gram.size(0),gram.size(1),gram.size(2),gram.size(3)*gram.size(4))
            gramNorm = torch.sqrt(torch.pow(gram*gram,2).sum(dim=-1))
            gramX = gram.unsqueeze(2)
            gramY = gram.unsqueeze(3)
            gramXNorm = gramNorm.unsqueeze(2)
            gramYNorm = gramNorm.unsqueeze(3)
            distMat = (torch.pow(gramX-gramY,2).sum(dim=-1)/(gramXNorm*gramYNorm))
            distMat_mean = distMat.mean(dim=1)

        return distMat_mean

class PatchSimCNN(torch.nn.Module):
    def __init__(self,resnet,resType,pretr,nbGroup,reluOnSimple,**kwargs):
        super(PatchSimCNN,self).__init__()

        self.resnet = resnet
        if not resnet:
            self.filterSizes = [3,5,7,11,15,23,37,55]
            self.layers = torch.nn.ModuleList()

            for filterSize in self.filterSizes:
                if reluOnSimple:
                    layer = torch.nn.Sequential(torch.nn.Conv2d(3,64,filterSize,padding=(filterSize-1)//2),torch.nn.ReLU())
                else:
                    layer = torch.nn.Conv2d(3,64,filterSize,padding=(filterSize-1)//2)
                self.layers.append(layer)
        else:
            self.featMod = modelBuilder.buildFeatModel(resType, pretr, True, False,**kwargs)

        self.nbGroup = nbGroup

    def forward(self,x):

        if not self.resnet:
            featList = []
            for i,layer in enumerate(self.layers):
                layerFeat = layer(x)
                featList.append(layerFeat)

            featVolume = torch.cat(featList,dim=1)
        else:
            featVolume = self.featMod(x)["x"]

        origFeatSize = featVolume.size()
        featVolume = featVolume.unfold(1, featVolume.size(1)//self.nbGroup, featVolume.size(1)//self.nbGroup).permute(0,1,4,2,3)
        featVolume = featVolume.view(featVolume.size(0)*featVolume.size(1),featVolume.size(2),featVolume.size(3),featVolume.size(4))

        gramMat = graham(featVolume)
        gramMat = gramMat.view(origFeatSize[0],self.nbGroup,origFeatSize[1]//self.nbGroup,origFeatSize[1]//self.nbGroup)

        return gramMat

class PxlSimCNN(torch.nn.Module):
    def __init__(self,resnet,pretr,kerSize,groupNb,stride,kerType):
        super(PxlSimCNN,self).__init__()

        self.resnet = resnet
        if not resnet:
            self.filterSizes = [3,5,7,11,15,23,37,55]
            self.layers = torch.nn.ModuleList()

            for filterSize in self.filterSizes:
                layer = torch.nn.Sequential(torch.nn.Conv2d(3,64,filterSize,padding=(filterSize-1)//2),torch.nn.ReLU())
                self.layers.append(layer)
        else:
            self.featMod = modelBuilder.buildFeatModel("resnet18", pretr, True, False)

        self.kernel = None
        self.kerSize = kerSize
        self.groupNb = groupNb
        self.stride = stride
        self.kerType = kerType
    def forward(self,x):

        if not self.resnet:
            featList = []
            for i,layer in enumerate(self.layers):
                layerFeat = layer(x)
                featList.append(layerFeat)

            featVolume = torch.cat(featList,dim=1)
        else:
            featVolume = self.featMod(x)["x"]

        origFeatSize = featVolume.size()


        print(featVolume.size())
        featVolume = featVolume.unfold(1, featVolume.size(1)//self.groupNb, featVolume.size(1)//self.groupNb).permute(0,1,4,2,3)
        print(featVolume.size())
        featVolume = featVolume.view(featVolume.size(0)*featVolume.size(1),featVolume.size(2),featVolume.size(3),featVolume.size(4))
        print(featVolume.size())
        if self.kernel is None:
            self.kernel = self.buildKernel(featVolume.size(1),self.kerType,featVolume.device)
        gramMatMap = grahamPxl(featVolume,self.kernel,self.stride)
        print(gramMatMap.size())
        simMap = modelBuilder.computeTotalSim(gramMatMap,1)
        print(simMap.size())
        simMap = simMap.reshape(x.size(0),self.groupNb,simMap.size(2),simMap.size(3))
        print(simMap.size())
        simMap = simMap.mean(dim=1,keepdim=True)
        print(simMap.size())

        return simMap

    def buildKernel(self,featNb,type,device):

        if type == "constant":
            kernel = torch.ones((featNb*featNb,1,self.kerSize,self.kerSize)).to(device)
            kernel /= self.kerSize*self.kerSize

        elif type == "linear" or type == "squarred":
            ordKer = (torch.arange(self.kerSize) - self.kerSize // 2).unsqueeze(1).unsqueeze(0).unsqueeze(0).expand(1, 1, self.kerSize, self.kerSize).float()
            absKer = (torch.arange(self.kerSize) - self.kerSize // 2).unsqueeze(0).unsqueeze(0).unsqueeze(0).expand(1, 1, self.kerSize, self.kerSize).float()
            ordKer, absKer = (ordKer.to(device), absKer.to(device))
            kernel = self.kerSize - (torch.abs(ordKer) + torch.abs(absKer))

            if type == "squarred":
                kernel = kernel*kernel

            kernel /= kernel.sum(dim=-1,keepdim=True).sum(dim=-1,keepdim=True)
            kernel = kernel.expand(featNb*featNb,-1,-1,-1)

        else:
            raise ValueError("Unkown kernel type : ",type)

        return kernel

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
    return modelBuilder.computeTotalSim(featCorr,1)

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

def computeTotalSim(features,dilRange=(1,6)):

    totalSim = None

    weightSum = np.arange(dilRange[0],dilRange[1])
    weightSum = (weightSum*weightSum).sum()

    for i in range(dilRange[0],dilRange[1]):
        sim = modelBuilder.computeTotalSim(features,i)

        weight = (i*i)/(weightSum)

        if totalSim is None:
            totalSim = sim*weight
        else:
            totalSim += sim*weight

    return totalSim

def computeLoss(reconst_weight,reconst_loss_module,text_enc_weight,neighpred_weight,gram_matrix_weight,gram_thres,gram_loss_module,\
                simclrDict,resDict, data, target,posDil,negDil,margin,layer):

    if layer<4:
        features = resDict["layerFeat"][layer]
    else:
        features = resDict["features"]

    totalSimPos = modelBuilder.computeTotalSim(features,posDil)
    totalSimNeg = modelBuilder.computeTotalSim(features,negDil)

    loss= -text_enc_weight*torch.max(totalSimPos.mean() - totalSimNeg.mean()+margin,0)[0]

    if reconst_weight > 0:
        reconst = resDict['reconst']
        loss += reconst_weight * reconst_loss_module(reconst,data).mean()
    else:
        reconst = None

    if neighpred_weight > 0:
        error_map = resDict["neighFeatPredErr"]
        loss += neighpred_weight*error_map.mean()

    graham_loss = 0
    if gram_matrix_weight != 0:

        for layer in resDict["layerFeat"].keys():
            if layer < 4:
                feat = resDict["layerFeat"][layer].detach()
                feat_decod = resDict["layerFeat_decod"][layer]

                selectedFilters = torch.randint(feat.size(1),size=(64,))

                feat = feat[:,selectedFilters]
                feat_decod = feat_decod[:,selectedFilters]

                graham_loss += gram_matrix_weight*gram_loss_module(feat,feat_decod).mean()

    if graham_loss > gram_thres and gram_matrix_weight > 0:
        total_loss = loss+graham_loss
    elif graham_loss < -gram_thres and gram_matrix_weight < 0:
        total_loss = loss+graham_loss
    else:
        total_loss = loss

    if not simclrDict is None:
        simCLR_loss = simclrDict["weight"]*simclrDict["loss"](simclrDict["z1"],simclrDict["z2"])
    else:
        simCLR_loss = 0

    total_loss += simCLR_loss

    return {"total_loss":total_loss,"loss":loss,"graham_loss":graham_loss,"simCLR_loss":simCLR_loss},totalSimPos,reconst

def graham(feat):
    feat = feat.reshape(feat.size(0),feat.size(1),feat.size(2)*feat.size(3))
    gram = (feat.unsqueeze(2)*feat.unsqueeze(1)).sum(dim=-1)
    return gram

def grahamPxl(feat,kernel,stride):
    #feat = feat.reshape(feat.size(0),feat.size(1),feat.size(2)*feat.size(3))
    gram = (feat.unsqueeze(2)*feat.unsqueeze(1))
    gram = gram.reshape(gram.size(0),gram.size(1)*gram.size(2),gram.size(3),gram.size(4))

    gram = torch.nn.functional.conv2d(gram,kernel,groups=feat.size(1)*feat.size(1),padding=(kernel.size(-1)-1)//2,stride=stride)

    return gram

def mask_correlated_samples(args):
    mask = torch.ones((args.batch_size * 2, args.batch_size * 2), dtype=bool)
    mask = mask.fill_diagonal_(0)
    for i in range(args.batch_size):
        mask[i, args.batch_size + i] = 0
        mask[args.batch_size + i, i] = 0
    return mask

if __name__ == "__main__":
    main()
