import os
import sys
import glob
import time

import numpy as np
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import torchvision
import torchvision.transforms as transforms
import captum 
import captum.attr

import args
from args import ArgReader
from args import str2bool
import modelBuilder
import load_data
import metrics
import utils
from road import NoisyLinearImputer,LinearInterpImputer
from trainVal import addInitArgs,preprocessAndLoadParams

def find_other_class_labels(inds,testDataset):

    inds_bckgr = (inds + len(testDataset)//2) % len(testDataset)

    labels = np.array([testDataset[ind][1] for ind in inds])
    labels_bckgr = np.array([testDataset[ind][1] for ind in inds_bckgr])

    while (labels==labels_bckgr).any():
        for i in range(len(inds)):
            if inds[i] == inds_bckgr[i]:
                inds_bckgr[i] = torch.randint(len(testDataset),size=(1,))[0]

        inds_bckgr = torch.randint(len(testDataset),size=(len(inds),))
        labels_bckgr = np.array([testDataset[ind][1] for ind in inds_bckgr])

    return inds_bckgr

#From https://kai760.medium.com/how-to-use-torch-fft-to-apply-a-high-pass-filter-to-an-image-61d01c752388
def roll_n(X, axis, n):
    f_idx = tuple(slice(None, None, None) 
            if i != axis else slice(0, n, None) 
            for i in range(X.dim()))
    b_idx = tuple(slice(None, None, None) 
            if i != axis else slice(n, None, None) 
            for i in range(X.dim()))
    front = X[f_idx]
    back = X[b_idx]
    return torch.cat([back, front], axis)
    
def fftshift(X):
    real, imag = X.chunk(chunks=2, dim=-1)
    real, imag = real.squeeze(dim=-1), imag.squeeze(dim=-1)    
    for dim in range(2, len(real.size())):
        real = roll_n(real, axis=dim,n=int(np.ceil(real.size(dim) / 2)))
        imag = roll_n(imag, axis=dim,n=int(np.ceil(imag.size(dim) / 2)))
    real, imag = real.unsqueeze(dim=-1), imag.unsqueeze(dim=-1)
    X = torch.cat((real,imag),dim=1)
    return torch.squeeze(X)
    
def ifftshift(X):
    real, imag = X.chunk(chunks=2, dim=-1)
    real, imag = real.squeeze(dim=-1), imag.squeeze(dim=-1)
    for dim in range(len(real.size()) - 1, 1, -1):
        real = roll_n(real, axis=dim,n=int(np.floor(real.size(dim) / 2)))
        imag = roll_n(imag, axis=dim,n=int(np.floor(imag.size(dim) / 2)))
    real, imag = real.unsqueeze(dim=-1), imag.unsqueeze(dim=-1)
    X = torch.cat((real, imag), dim=1)
    return torch.squeeze(X)

def maskData(data,attMaps_interp,data_bckgr):
    data_masked = data*attMaps_interp+data_bckgr*(1-attMaps_interp)
    return data_masked 

def getBatch(testDataset,i,args):
    batch = testDataset.__getitem__(i)
    data,targ = batch[0].unsqueeze(0),torch.tensor(batch[1]).unsqueeze(0)
    data = data.cuda() if args.cuda else data
    targ = targ.cuda() if args.cuda else targ
    return data,targ 

def computeSpars(data_shape,attMaps,args,resDic):
    if args.att_metrics_post_hoc:
        features = None 
    else:
        features = resDic["feat"]
        if "attMaps" in resDic:
            attMaps = resDic["attMaps"]
        else:
            attMaps = torch.ones(data_shape[0],1,features.size(2),features.size(3)).to(features.device)

    sparsity = metrics.compAttMapSparsity(attMaps,features)
    sparsity = sparsity/data_shape[0]
    return sparsity 

def inference(net,data,predClassInd,args):
    if not args.prototree:
        resDic = net(data)
        score = torch.softmax(resDic["pred"],dim=-1)[:,predClassInd[0]]
        feat = resDic["feat_pooled"]
    else:
        score = net(data)[0][:,predClassInd[0]]
        feat = None
    return score,feat

def applyPostHoc(attrFunc,data,targ,kwargs,args):

    attMaps = []
    for i in range(len(data)):

        if args.att_metrics_post_hoc.find("var") == -1 and args.att_metrics_post_hoc.find("smooth") == -1:
            argList = [data[i:i+1],targ[i:i+1]]
        else:
            argList = [data[i:i+1]]
            kwargs["target"] = targ[i:i+1]

        attMap = attrFunc(*argList,**kwargs).clone().detach().to(data.device)

        if len(attMap.size()) == 2:
            attMap = attMap.unsqueeze(0).unsqueeze(0)
        elif len(attMap.size()) == 3:
            attMap = attMap.unsqueeze(0)
        
        attMaps.append(attMap)
    
    return torch.cat(attMaps,dim=0)
    
def getAttMetrMod(net,testDataset,args):
    if args.att_metrics_post_hoc == "gradcam":
        netGradMod = modelBuilder.GradCamMod(net)
        attrMod = captum.attr.LayerGradCam(netGradMod.forward,netGradMod.layer4)
        attrFunc = attrMod.attribute
        kwargs = {}
    elif args.att_metrics_post_hoc == "gradcam_pp":
        netGradMod = modelBuilder.GradCamMod(net)
        model_dict = dict(type=args.first_mod, arch=netGradMod, layer_name='layer4', input_size=(448, 448))
        attrMod = GradCAMpp(model_dict,True)
        attrFunc = attrMod.__call__
        kwargs = {}
    elif args.att_metrics_post_hoc == "guided":
        netGradMod = modelBuilder.GradCamMod(net)
        attrMod = captum.attr.GuidedBackprop(netGradMod)
        attrFunc = attrMod.attribute
        kwargs = {}
    elif args.att_metrics_post_hoc == "xgradcam":
        netGradMod = modelBuilder.GradCamMod(net)
        attrMod = XGradCAM(model=netGradMod,target_layers=netGradMod.layer4,use_cuda=args.cuda)
        attrFunc = attrMod
        kwargs = {}
    elif args.att_metrics_post_hoc == "ablation_cam":
        netGradMod = modelBuilder.GradCamMod(net)
        attrMod = AblationCAM(model=netGradMod,target_layers=netGradMod.layer4,use_cuda=args.cuda)
        attrFunc = attrMod
        kwargs = {}
    elif args.att_metrics_post_hoc == "score_map":
        attrMod = ScoreCam(net)
        attrFunc = attrMod.generate_cam
        kwargs = {}
    elif args.att_metrics_post_hoc == "varGrad" or args.att_metrics_post_hoc == "smoothGrad":
        torch.backends.cudnn.benchmark = False
        torch.set_grad_enabled(False)
        
        net.eval()
        netGradMod = modelBuilder.GradCamMod(net)

        ig = IntegratedGradients(netGradMod)
        attrMod = NoiseTunnel(ig)
        attrFunc = attrMod.attribute

        batch = testDataset.__getitem__(0)
        data_base = torch.zeros_like(batch[0].unsqueeze(0))

        if args.cuda:
            data_base = data_base.cuda()
        kwargs = {"nt_type":'smoothgrad_sq' if args.att_metrics_post_hoc == "smoothGrad" else "vargrad", \
                        "stdevs":0.02, "nt_samples":3,"nt_samples_batch_size":3}
    elif args.att_metrics_post_hoc == "rise":
        torch.set_grad_enabled(False)
        attrMod = RISE(net)
        attrFunc = attrMod.__call__
        kwargs = {}
    else:
        raise ValueError("Unknown post-hoc method",args.att_metrics_post_hoc)
    return attrFunc,kwargs

def loadAttMaps(exp_id,model_id):

    paths = glob.glob("../results/{}/attMaps_{}_epoch*.npy".format(exp_id,model_id))

    if len(paths) >1 or len(paths) == 0:
        raise ValueError("Wrong path number for exp {} model {}",exp_id,model_id)

    attMaps,norm = np.load(paths[0],mmap_mode="r"),np.load(paths[0].replace("attMaps","norm"),mmap_mode="r")

    return torch.tensor(attMaps),torch.tensor(norm)

def main(argv=None):
    # Getting arguments from config file and command line
    # Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--attention_metrics', type=str, help='The attention metric to compute.')
    argreader.parser.add_argument('--att_metrics_img_nb', type=int, help='The nb of images on which to compute the att metric.')
    
    argreader.parser.add_argument('--att_metrics_map_resolution', type=int, help='The resolution at which to resize the attention map.\
                                    If this value is None, the map will not be resized.')
    argreader.parser.add_argument('--att_metrics_sparsity_factor', type=float, help='Used to increase (>1) or decrease (<1) the sparsity of the saliency maps.\
                                    Set to None to not modify the sparsity.')

    argreader.parser.add_argument('--att_metrics_post_hoc', type=str, help='The post-hoc method to use instead of the model ')
    argreader.parser.add_argument('--att_metrics_max_brnpa', type=str2bool, help='To agregate br-npa maps with max instead of mean')
    argreader.parser.add_argument('--att_metrics_onlyfirst_brnpa', type=str2bool, help='To agregate br-npa maps with max instead of mean')
    argreader.parser.add_argument('--att_metrics_few_steps', type=str2bool, help='To do as much step for high res than for low res')
    argreader.parser.add_argument('--att_metr_do_again', type=str2bool, help='To run computation if already done',default=True)

    argreader.parser.add_argument('--att_metr_bckgr', type=str, help='The pixel replacement method.',default="black")
        
    argreader.parser.add_argument('--att_metr_save_feat', type=str2bool, help='',default=False)
    argreader.parser.add_argument('--att_metr_add_first_feat', type=str2bool, help='',default=False)

    argreader = addInitArgs(argreader)

    argreader = modelBuilder.addArgs(argreader)
    argreader = load_data.addArgs(argreader)

    # Reading the comand line arg
    argreader.getRemainingArgs()

    args = argreader.args
  
    path_suff = args.attention_metrics
    path_suff += "-"+args.att_metr_bckgr if args.att_metr_bckgr != "black" else ""
            
    model_id_suff = "-"+args.att_metrics_post_hoc if args.att_metrics_post_hoc else ""
    model_id_suff += "-res"+str(args.att_metrics_map_resolution) if args.att_metrics_map_resolution else ""
    model_id_suff += "-spar"+str(args.att_metrics_sparsity_factor) if args.att_metrics_sparsity_factor else ""

    score_path = "../results/{}/attMetr{}_{}{}.npy".format(args.exp_id,path_suff,args.model_id,model_id_suff)
    feat_path = "../results/{}/attMetrFeat{}_{}{}.npy".format(args.exp_id,path_suff,args.model_id,model_id_suff)

    if args.att_metr_do_again or (not os.path.exists(score_path)) or ((not os.path.exists(feat_path)) and args.att_metr_save_feat):

        args.val_batch_size = 1
        testLoader,testDataset = load_data.buildTestLoader(args, "test")

        #Useful to generate gray and white background
        mean,std = torch.tensor([0.485, 0.456, 0.406]), torch.tensor([0.229, 0.224, 0.225])
        denormalize = transforms.Compose([transforms.Normalize(mean=[0.,0.,0.],std=1/std),\
                                            transforms.Normalize(mean =-mean,std=[1,1,1])])

        bestPath = glob.glob("../models/{}/model{}_best_epoch*".format(args.exp_id, args.model_id))[0]
        bestEpoch = int(os.path.basename(bestPath).split("epoch")[1])

        net = modelBuilder.netBuilder(args)
        net = preprocessAndLoadParams(bestPath,args.cuda,net)
        net.eval()
        
        if args.att_metrics_post_hoc:
            attrFunc,kwargs = getAttMetrMod(net,testDataset,args)
        else:
            attMaps_dataset,norm_dataset = loadAttMaps(args.exp_id,args.model_id)

            if not args.resnet_bilinear or (args.resnet_bilinear and args.bil_cluster):
                attrFunc = lambda i:(attMaps_dataset[i,0:1]*norm_dataset[i]).unsqueeze(0)
            else:
                attrFunc = lambda i:(attMaps_dataset[i].float().mean(dim=0,keepdim=True).byte()*norm_dataset[i]).unsqueeze(0)
        
        if args.att_metrics_post_hoc != "gradcam_pp":
            torch.set_grad_enabled(False)

        nbImgs = args.att_metrics_img_nb

        if args.attention_metrics in ["Del","Add"]:
            allScoreList = []
            allPreds = []
            allTarg = []
            allFeat = []
        elif args.attention_metrics == "AttScore":
            allAttScor = []
        elif args.attention_metrics == "Time":
            allTimeList = []
        elif args.attention_metrics == "Lift":
            allScoreList = []
            allScoreMaskList = []
            allScoreInvMaskList = []
            allFeat = []
        else:
            allSpars = []

        torch.manual_seed(0)
        inds = torch.randint(len(testDataset),size=(nbImgs,))

        if args.att_metr_bckgr=="IB":
            inds_bckgr = find_other_class_labels(inds,testDataset)
                
        for imgInd,i in enumerate(inds):
            if imgInd % 20 == 0 :
                print("Img",i.item(),"(",imgInd,"/",len(inds),")")

            data,targ = getBatch(testDataset,i,args)
            if args.dataset_test.find("emb") == -1:
                data_unorm = denormalize(data)
            else:
                data_unorm = data
            allData = data_unorm.clone().cpu()

            startTime = time.time()
            resDic = net(data)
            scores = torch.softmax(resDic["pred"],dim=-1)
            inf_time = time.time() - startTime

            if args.attention_metrics in ["Add","Del"]:
                predClassInd = scores.argmax(dim=-1)
                allPreds.append(predClassInd.item())
                allTarg.append(targ.item())

            if args.att_metrics_post_hoc:
                startTime = time.time()
                attMaps = applyPostHoc(attrFunc,data,targ,kwargs,args)
                totalTime = inf_time + time.time() - startTime
            else:
                attMaps = attrFunc(i)
                totalTime = inf_time

            if args.att_metr_bckgr=="IB":
                data_bckgr,_ = getBatch(testDataset,inds_bckgr[imgInd],args)
            elif args.att_metr_bckgr == "black":
                data_bckgr = torch.zeros_like(data)
            elif args.att_metr_bckgr == "road":
                data_bckgr = torch.zeros_like(data)
                imputer = NoisyLinearImputer()
            elif args.att_metr_bckgr == "linear":
                data_bckgr = torch.zeros_like(data)
                imputer = LinearInterpImputer()
            elif args.att_metr_bckgr == "white":
                data_bckgr = torch.ones_like(data)
            elif args.att_metr_bckgr == "gray":
                data_bckgr = 0.5*torch.ones_like(data)
            elif args.att_metr_bckgr in ["lowpass","highpass"]:

                cutoff_freq = np.genfromtxt("../results/EMB10/cutoff_filter_freq.csv")

                fft = torch.fft.fft2(data[0,0])
                fft = np.fft.fftshift(fft.cpu())

                cx,cy = fft.shape[1]//2,fft.shape[0]//2
                rx,ry = int(cutoff_freq*cx*0.5),int(cutoff_freq*cy*0.5)

                mask = np.ones_like(fft,dtype="float64")
                mask[cy-ry:cy+ry,cx-rx:cx+rx] = 0

                if args.att_metr_bckgr == "highpass":
                    mask = 1 - mask
                
                fft[mask.astype("bool")] = 0

                fft = np.fft.ifftshift(fft)
                data_bckgr = torch.fft.ifft2(torch.tensor(fft).to(data.device)).real.unsqueeze(0).unsqueeze(0)

                data_bckgr = data_bckgr.expand(-1,data.size(1),-1,-1)
            elif args.att_metr_bckgr == "blur":
                kernel = torch.ones(121,121)
                kernel = kernel/kernel.numel()
                kernel = kernel.unsqueeze(0).unsqueeze(0).expand(3,1,-1,-1)
                kernel = kernel.cuda() if args.cuda else kernel
                data_bckgr = F.conv2d(data,kernel,padding=kernel.size(-1)//2,groups=kernel.size(0))            
            else:
                raise ValueError("Unkown background method",args.att_metr_bckgr)

            if args.attention_metrics=="Add":
                origData = data.clone()
                origUnormData = data_unorm.clone()
                data = data_bckgr.clone()
                if args.att_metr_bckgr in ["IB","edge","blur","highpass","lowpass"] and args.dataset_test.find("emb") == -1:
                    data_unorm = denormalize(data_bckgr.clone())
                else:
                    data_unorm = data_bckgr.clone()
                first_sample_feat = net(data)["x"]
            elif args.attention_metrics == "Del":
                first_sample_feat = resDic["feat_pooled"]

            attMaps = (attMaps-attMaps.min())/(attMaps.max()-attMaps.min())

            if args.att_metrics_map_resolution:
                attMaps = torch.nn.functional.interpolate(attMaps,size=args.att_metrics_map_resolution,mode="bicubic",align_corners=False).to(data.device)                    

            if args.att_metrics_sparsity_factor and (args.att_metrics_sparsity_factor!= 1):
                attMaps = torch.pow(attMaps,args.att_metrics_sparsity_factor)
                attMaps = (attMaps-attMaps.min())/(attMaps.max()-attMaps.min())

            if args.attention_metrics=="Spars":
                sparsity= computeSpars(data.size(),attMaps,args,resDic)
                allSpars.append(sparsity)
            elif args.attention_metrics == "AttScore":
                allAttScor.append(attMaps.view(-1).sort()[0].detach().cpu().numpy())
            elif args.attention_metrics == "Time":
                allTimeList.append(totalTime)
            elif args.attention_metrics == "Lift":
                predClassInd = scores.argmax(dim=-1)
                score = scores[:,predClassInd[0]].cpu().detach().numpy()
                allScoreList.append(score)
                interp_mode = "nearest" if args.att_metrics_map_resolution else "bicubic"
                attMaps_interp = torch.nn.functional.interpolate(attMaps,size=(data.shape[-1]),mode=interp_mode).to(data.device)                    
                
                feat = resDic["feat_pooled"].detach().cpu()

                data_masked = maskData(data,attMaps_interp,data_bckgr)
                data_unorm_masked = maskData(data_unorm,attMaps_interp,data_bckgr)
                allData = torch.cat((allData,data_unorm_masked.cpu()),dim=0)
                score_mask,feat_mask = inference(net,data_masked,predClassInd,args)
                feat_mask = feat_mask.detach().cpu()
                allScoreMaskList.append(score_mask.cpu().detach().numpy())
                
                data_invmasked = maskData(data,1-attMaps_interp,data_bckgr)
                data_unorm_invmasked = maskData(data_unorm,1-attMaps_interp,data_bckgr)
                allData = torch.cat((allData,data_unorm_invmasked.cpu()),dim=0)
                score_invmask,feat_invmask = inference(net,data_invmasked,predClassInd,args)
                feat_invmask = feat_invmask.detach().cpu()
                allScoreInvMaskList.append(score_invmask.cpu().detach().numpy())

                allFeat.append(torch.cat((feat,feat_mask,feat_invmask),dim=0).unsqueeze(0))

            elif args.attention_metrics in ["Del","Add"]:
                allAttMaps = attMaps.clone().cpu()
                statsList = []

                totalPxlNb = attMaps.size(2)*attMaps.size(3)
                leftPxlNb = totalPxlNb

                if args.prototree or args.protonet:
                    stepNb = 49
                elif args.att_metrics_few_steps:
                    stepNb = min(196,attMaps.shape[-2]*attMaps.shape[-1])
                else:
                    stepNb = totalPxlNb

                score_prop_list = []

                ratio = data.size(-1)//attMaps.size(-1)

                stepCount = 0

                allFeatIter = [first_sample_feat.detach().cpu()]

                if not args.att_metr_add_first_feat:

                    while leftPxlNb > 0:

                        if args.att_metr_bckgr in ["road"]:
                            mask_shape = (data.shape[0],1,data.shape[2],data.shape[3])
                            imputer_mask = torch.zeros(mask_shape) if args.attention_metrics=="Add" else torch.ones(mask_shape)

                        attMin,attMean,attMax = attMaps.min().item(),attMaps.mean().item(),attMaps.max().item()
                        statsList.append((attMin,attMean,attMax))

                        _,ind_max = (attMaps)[0,0].view(-1).topk(k=totalPxlNb//stepNb)
                        ind_max = ind_max[:leftPxlNb]

                        x_max,y_max = ind_max % attMaps.shape[3],torch.div(ind_max,attMaps.shape[3],rounding_mode="floor")
                        
                        ratio = data.size(-1)//attMaps.size(-1)

                        for i in range(len(x_max)):
                            
                            x1,y1 = x_max[i]*ratio,y_max[i]*ratio,
                            x2,y2 = x1+ratio,y1+ratio

                            if args.attention_metrics=="Add":
                                data[0,:,y1:y2,x1:x2] = origData[0,:,y1:y2,x1:x2]
                                data_unorm[0,:,y1:y2,x1:x2] = origUnormData[0,:,y1:y2,x1:x2]
                            else:
                                data[0,:,y1:y2,x1:x2] = data_bckgr[0,:,y1:y2,x1:x2]
                                data_unorm[0,:,y1:y2,x1:x2] = data_bckgr[0,:,y1:y2,x1:x2]

                            attMaps[0,:,y_max[i],x_max[i]] = -1                       

                            if args.att_metr_bckgr == "road":
                                if args.attention_metrics=="Add":
                                    imputer_mask[0,0,y1:y2,x1:x2] = 1
                                else:
                                    imputer_mask[0,0,y1:y2,x1:x2] = 0

                        if args.att_metr_bckgr in ["road"]:
                            data = imputer(data.cpu(),imputer_mask)
                        elif args.att_metr_bckgr == "linear":
                            data = imputer(data.cpu(),x1,x2-1,y1,y2-1)

                        leftPxlNb -= totalPxlNb//stepNb
                        if stepCount % 25 == 0:
                            allAttMaps = torch.cat((allAttMaps,torch.clamp(attMaps,0,attMaps.max().item()).cpu()),dim=0)
                            allData = torch.cat((allData,data_unorm.cpu().clone()),dim=0)
                        
                        stepCount += 1

                        score,feat = inference(net,data,predClassInd,args)

                        allFeatIter.append(feat.detach().cpu())

                        score_prop_list.append((leftPxlNb,score.item()))

                allFeat.append(torch.cat(allFeatIter,dim=0).unsqueeze(0))
                allScoreList.append(score_prop_list)

            else:
                raise ValueError("Unkown attention metric",args.attention_metrics)
        if not args.att_metr_add_first_feat:
            if args.attention_metrics == "Spars":
                np.save("../results/{}/attMetrSpars_{}{}.npy".format(args.exp_id,args.model_id,model_id_suff),np.array(allSpars,dtype=object))
            elif args.attention_metrics == "AttScore":
                np.save("../results/{}/attMetrAttScore_{}{}.npy".format(args.exp_id,args.model_id,model_id_suff),np.array(allAttScor,dtype=object))
            elif args.attention_metrics == "Time":
                np.save("../results/{}/attMetrTime_{}{}.npy".format(args.exp_id,args.model_id,model_id_suff),np.array(allTimeList,dtype=object))
            elif args.attention_metrics == "Lift":
                suff = path_suff.replace("Lift","")
                np.save("../results/{}/attMetrLift{}_{}{}.npy".format(args.exp_id,suff,args.model_id,model_id_suff),np.array(allScoreList,dtype=object))
                np.save("../results/{}/attMetrLiftMask{}_{}{}.npy".format(args.exp_id,suff,args.model_id,model_id_suff),np.array(allScoreMaskList,dtype=object))
                np.save("../results/{}/attMetrLiftInvMask{}_{}{}.npy".format(args.exp_id,suff,args.model_id,model_id_suff),np.array(allScoreInvMaskList,dtype=object))
            else:
                np.save(score_path,np.array(allScoreList,dtype=object))
                np.save("../results/{}/attMetrPreds{}_{}{}.npy".format(args.exp_id,path_suff,args.model_id,model_id_suff),np.array(allPreds,dtype=object))
    
            if args.attention_metrics in ["Lift","Del","Add"] and args.att_metr_save_feat:
                allFeat = torch.cat(allFeat,dim=0)
                np.save(feat_path,allFeat.numpy())

            allDataPath = "../vis/{}/attMetrData{}_{}{}.png".format(args.exp_id,path_suff,args.model_id,model_id_suff)
            nrows = 1  if args.attention_metrics == "Lift" else 4
            torchvision.utils.save_image(allData,allDataPath,nrow=nrows)
        else:
            allFeat = torch.cat(allFeat,dim=0)
            featPath = f"../results/{args.exp_id}/attMetrFeat{path_suff}_{args.model_id}{model_id_suff}.npy"  
            allFeat = np.concatenate((allFeat.numpy(),np.load(featPath,mmap_mode="r")),axis=1)
            np.save(featPath,allFeat)

if __name__ == "__main__":
    main()