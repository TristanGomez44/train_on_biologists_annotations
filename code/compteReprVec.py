import torch
import args
import modelBuilder
import load_data
import testOneImageTextAnalysis
import trainVal
from args import ArgReader
import time
import numpy as np
import umap
import cuml
def main(argv=None):
    # Getting arguments from config file and command line
    # Building the arg reader
    argreader = ArgReader(argv)

    argreader = modelBuilder.addArgs(argreader)
    argreader = load_data.addArgs(argreader)
    argreader = testOneImageTextAnalysis.addArgs(argreader)

    # Reading the comand line arg
    argreader.getRemainingArgs()

    args = argreader.args

    #trainLoader, _ = load_data.buildTrainLoader(args,transf=transf,shuffle=args.shuffle)
    args.train_prop = 100
    trainLoader, _ = load_data.buildTrainLoader(args,transf="identity",shuffle=False)
    testLoader,_ = load_data.buildTestLoader(args, "test")

    cosimMap = buildModule(True,CosimMap,args.cuda,args.multi_gpu,{})

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

    kwargsNet = {"resnet":args.patch_sim_resnet,"resType":args.patch_sim_restype,"pretr":args.patch_sim_pretr_res,"nbGroup":args.patch_sim_group_nb,\
                "reluOnSimple":args.patch_sim_relu_on_simple,"chan":args.resnet_chan,"gramOrder":args.patch_sim_gram_order,"useModel":args.patch_sim_usemodel,\
                "inChan":3,"resnet_multilev":args.resnet_multilev}
    patchMod = buildModule(True,PatchSimCNN,args.cuda,args.multi_gpu,kwargsNet)

    for j,dataLoader in enumerate([trainLoader,testLoader]):
        allReprVec = None
        print("Dataset",j)
        for i,(data,target) in enumerate(dataLoader):

            with torch.no_grad():

                data = data.cuda() if args.cuda else data

                patch = data.unfold(2, args.patch_size, args.patch_stride).unfold(3, args.patch_size, args.patch_stride).permute(0,2,3,1,4,5)
                patch = patch.reshape(patch.size(0)*patch.size(1)*patch.size(2),patch.size(3),patch.size(4),patch.size(5))

                reprVec = textLimit(patch,patchMod,data,cosimMap,neighSimMod,representativeVectorsMod,kwargsNet,args)

                if allReprVec is None:
                    allReprVec = reprVec
                else:
                    allReprVec = torch.cat((allReprVec,reprVec),dim=0)

            if i % 10 == 0:
                print("\t",i,data.size(0)*i)

        np.save("../results/{}_reprVec.npy".format([args.dataset_train,args.dataset_test][j]),allReprVec.cpu().numpy())

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

def computeNeighborsCoord(neighRadius):

    coord = torch.arange(neighRadius*2+1)-neighRadius

    y = coord.unsqueeze(1).expand(coord.size(0),coord.size(0)).unsqueeze(-1)
    x = coord.unsqueeze(0).expand(coord.size(0),coord.size(0)).unsqueeze(-1)

    coord = torch.cat((x,y),dim=-1).view(coord.size(0)*coord.size(0),2)
    coord = coord[~((coord[:,0] == 0)*(coord[:,1] == 0))]

    return coord

def textLimit(patch,patchMod,data,cosimMap,neighSimMod,representativeVectorsMod,kwargsNet,args):

    gram_mat = patchMod(patch)
    gram_mat = gram_mat.view(data.size(0),-1,args.patch_sim_group_nb,gram_mat.size(2))
    gram_mat = gram_mat.permute(0,2,1,3)
    feat = cosimMap(gram_mat)

    neighSim,refFeatList = neighSimMod(feat)

    reprVec,simListRepr,selectedPos = representativeVectorsMod(refFeatList[-1],applyUMAP=True)

    return reprVec

def representativeVectors(xRaw,nbVec,prior=None,unifSample=False,onlySimSample=False,applyUMAP=True,umapNeiNb=15,umapMinDist=0.1):

    if applyUMAP:
        emb_list = []
        for i in range(len(xRaw)):
            #emb = umap.UMAP(n_components=2,random_state=1,n_neighbors=umapNeiNb,min_dist=umapMinDist).fit_transform(xRaw[i].reshape(xRaw[i].size(0),-1).permute(1,0).cpu().detach().numpy())
            emb = cuml.UMAP(n_components=2,n_neighbors=umapNeiNb,min_dist=umapMinDist).fit_transform(xRaw[i].reshape(xRaw[i].size(0),-1).permute(1,0).contiguous().cpu().detach().numpy())
            emb = torch.tensor(emb).to(xRaw.device).reshape(xRaw[i].size(-2),xRaw[i].size(-1),-1).permute(2,0,1).unsqueeze(0)
            emb_list.append(emb)
        x = torch.cat(emb_list,dim=0)
    else:
        x = xRaw

    xOrigShape = x.size()
    normNotFlat = torch.sqrt(torch.pow(x,2).sum(dim=1,keepdim=True))
    x = x.permute(0,2,3,1).reshape(x.size(0),x.size(2)*x.size(3),x.size(1))
    xRaw = xRaw.permute(0,2,3,1).reshape(xRaw.size(0),xRaw.size(2)*xRaw.size(3),xRaw.size(1))
    norm = torch.sqrt(torch.pow(x,2).sum(dim=-1)) + 0.00001

    selectedPos = torch.zeros(xOrigShape[0],xOrigShape[2]*xOrigShape[3]).to(x.device)

    if prior is None:
        raw_reprVec_score = torch.rand(norm.size()).to(norm.device)
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
            raise ValueError("Choose one of unifSample or onlySimSample.")

        raw_reprVec = x[torch.arange(x.size(0)).unsqueeze(1),ind]

        sim = torch.exp(-torch.sqrt(torch.pow(x-raw_reprVec,2).sum(dim=-1))/20)

        simNorm = sim/sim.sum(dim=1,keepdim=True)

        reprVec = (xRaw*simNorm.unsqueeze(-1)).sum(dim=1)
        repreVecList.append(reprVec)

        if onlySimSample and i == 0:
            raw_reprVec_score = (1-sim)
        else:
            raw_reprVec_score = (1-sim)*raw_reprVec_score

        simReshaped = sim.reshape(sim.size(0),1,xOrigShape[2],xOrigShape[3])

        simList.append(simReshaped)

    selectedPos = selectedPos.reshape(selectedPos.size(0),1,xOrigShape[2],xOrigShape[3])

    return repreVecList,simList,selectedPos

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

def computeTotalSim(features,dilation,neigAvgSize=1,reprVec=False,centered=False,eucli=False):
    horizDiff,_,_,_,_ = applyDiffKer_CosSimi("horizontal",features,dilation,neigAvgSize,reprVec,centered,eucli)
    vertiDiff,_,_,_,_ = applyDiffKer_CosSimi("vertical",features,dilation,neigAvgSize,reprVec,centered,eucli)
    totalDiff = (horizDiff + vertiDiff)/2
    return totalDiff

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

class CosimMap(torch.nn.Module):
    def __init__(self):
        super(CosimMap,self).__init__()

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
        return feat

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

if __name__ == "__main__":
    main()
