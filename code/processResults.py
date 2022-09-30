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
import umap
from PIL import ImageFont,Image,ImageDraw
from sklearn.manifold import TSNE
import sklearn
from sklearn import svm ,neural_network,tree,neighbors
from sklearn.manifold import  TSNE
from skimage.transform import resize
from scipy.stats import kendalltau
import load_data

def compNorm(featPath):

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
        print(feat.shape)
    return allNorm

def find_saliency_maps(viz_id,model_ids,exp_id,expl):
    mapPaths = []
    suff = "" if viz_id == "" else "{}_".format(viz_id)
    for j in range(len(model_ids)):
        print(f"../results/{exp_id}/{expl[j]}_{model_ids[j]}_epoch*_{suff}test.npy")
        paths = glob.glob(f"../results/{exp_id}/{expl[j]}_{model_ids[j]}_epoch*_{suff}test.npy")

        if len(paths) != 1:
            raise ValueError("Wrong paths number",paths)

        mapPaths.append(paths[0])
    return mapPaths 

def select_images(args,mapPaths,class_index,ind_to_keep,img_nb):
    _,testDataset = load_data.buildTestLoader(args,"test",shuffle=False)
    maxInd = len(glob.glob(os.path.join(args.dataset_test,"*/*.*")))

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
    inds = torch.randint(startInd,endInd,size=(img_nb,))

    if not ind_to_keep is None:
        ind_to_keep = np.array(ind_to_keep)-1
        inds = inds[ind_to_keep]

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

def showSalMaps(exp_id,img_nb,plot_id,nrows,class_index,ind_to_keep,viz_id,args,
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
    inds,imgBatch = select_images(args,mapPaths,class_index,ind_to_keep,img_nb)
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
            imgDraw.text((0,0), str(i+1)+" ", font=fnt,fill=(0,0,0))
            img = torch.tensor(np.array(imgPIL)).permute(2,0,1).unsqueeze(0).float()/255

        if gridImage is None:
            gridImage = img
        else:
            gridImage = torch.cat((gridImage,img),dim=0)

        for j in range(len(mapPaths)):
            
            all_attMaps = np.load(mapPaths[j],mmap_mode="r")
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

def get_metrics_with_bckgr():
    return ["Del","DelCorr","Add","AddCorr","AD","ADD","IIC","Lift"]

def get_bckgr_suff(bckgr):
    return "-"+bckgr if bckgr != "black" else ""

def get_resolution_suff(resolution):
    return "-res" if resolution else ""

def filter_background(paths,bckgr):
    bckgr_list = get_backgr_list()
    bckgr_suff = get_bckgr_suff(bckgr)

    if bckgr == "black":
        for i in range(len(bckgr_list)):
            if bckgr_list[i] != bckgr_suff:
                bckgr_suff_i = get_bckgr_suff(bckgr_list[i])
                paths = list(filter(lambda x:os.path.basename(x).split("-res")[0].find(bckgr_suff_i) ==-1,paths)) 
    else:
        paths = list(filter(lambda x:os.path.basename(x).split("-res")[0].find(bckgr_suff) !=-1,paths)) 

    print(len(paths))
    return paths

def getResPaths(exp_id,metric,bckgr):
    if metric in get_metrics_with_bckgr():
        bckgr_suff = get_bckgr_suff(bckgr)
        paths = sorted(glob.glob(f"../results/{exp_id}/attMetr{metric}{bckgr_suff}_*.npy"))
        paths = filter_background(paths,bckgr)
    else:
        paths = sorted(glob.glob(f"../results/{exp_id}/attMetr{metric}_*.npy"))
    return paths 

def getModelId(path,metric,img_bckgr):
    bckgr_suff = get_bckgr_suff(img_bckgr) if metric != "Spars" else ""
    #res_suff = get_resolution_suff(resolution)
    model_id = os.path.basename(path).split("attMetr{}{}_".format(metric,bckgr_suff))[1].split(".npy")[0]
    #if resolution:
    #    model_id = model_id.split(res_suff)[0]
    return model_id

def attMetrics(exp_id,metric,img_bckgr):

    suff = metric
    paths = getResPaths(exp_id,metric,img_bckgr)
    resDic = {}
    resDic_pop = {}

    if metric in ["Del","Add"]:
        for path in paths:

            model_id = getModelId(path,metric,img_bckgr)
            pairs = np.load(path,allow_pickle=True)

            allAuC = []

            for i in range(len(pairs)):

                pairs_i = np.array(pairs[i])

                if metric == "Add":
                    pairs_i[:,0] = 1-pairs_i[:,0]/pairs_i[:,0].max()
                else:
                    pairs_i[:,0] = (pairs_i[:,0]-pairs_i[:,0].min())/(pairs_i[:,0].max()-pairs_i[:,0].min())
                    pairs_i[:,0] = 1-pairs_i[:,0]

                auc = np.trapz(pairs_i[:,1],pairs_i[:,0])
                allAuC.append(auc)
        
            resDic_pop[model_id] = np.array(allAuC)
            resDic[model_id] = resDic_pop[model_id].mean()
    elif metric == "Lift":

        for path in paths:

            model_id = getModelId(path,metric,img_bckgr)
                            
            scores = np.load(path,allow_pickle=True)[:,0] 
            scores_mask = np.load(path.replace("Lift","LiftMask"),allow_pickle=True)[:,0] 
            scores_invmask = np.load(path.replace("Lift","LiftInvMask"),allow_pickle=True)[:,0]  

            iic = 100*(scores<scores_mask).mean(keepdims=True)
            diff = (scores-scores_mask)
            ad = 100*diff*(diff>0)/scores
            add = 100*(scores-scores_invmask)/scores

            resDic[model_id] = str(iic.item())+","+str(ad.mean())+","+str(add.mean())
            resDic_pop[model_id] = {"IIC":iic,"AD":ad,"ADD":add}
    else:
        for path in paths:
            
            model_id = getModelId(path,metric,img_bckgr)
            
            sparsity_list = 1/np.load(path,allow_pickle=True)

            resDic_pop[model_id] = np.array(sparsity_list)
            resDic[model_id] = resDic_pop[model_id].mean() 


    bckgr_suff = get_bckgr_suff(img_bckgr) if metric != "Spars" else ""
    #res_suff = get_resolution_suff(resolution) if metric != "Spars" else ""
    #suff += bckgr_suff+res_suff
    suff = bckgr_suff

    csv = "\n".join(["{},{}".format(key,resDic[key]) for key in resDic])
    with open(f"../results/{exp_id}/attMetrics_{metric}{suff}.csv","w") as file:
        print(csv,file=file)

    if metric == "Lift":
        for metric in get_lift_submetric():
            csv = "\n".join(["{},{}".format(key,",".join(resDic_pop[key][metric].astype("str"))) for key in resDic_pop])
            with open("../results/{}/attMetrics_{}{}_pop.csv".format(exp_id,metric,suff),"w") as file:
                print(csv,file=file)

    else:
        csv = "\n".join(["{},{}".format(key,",".join(resDic_pop[key].astype("str"))) for key in resDic_pop])
        with open("../results/{}/attMetrics_{}_pop.csv".format(exp_id,metric,suff),"w") as file:
            print(csv,file=file)

def get_lift_submetric():
    return ["IIC","AD","ADD"]

def get_id_to_label():
    return {"bilRed":"B-CNN",
            "bilRed_1map":"B-CNN (1 map)",
            "clus_masterClusRed":"BR-NPA",
            "clus_mast":"BR-NPA",
            "noneRed":"AM",
            "protopnet":"ProtoPNet",
            "prototree":"ProtoTree",
            "noneRed-gradcam":"Grad-CAM",
            "noneRed-gradcam_pp":"Grad-CAM++",
            "noneRed-score_map":"Score-CAM",
            "noneRed-ablation_cam":"Ablation-CAM",
            "noneRed-rise":"RISE",
            "noneRed_smallimg-varGrad":"VarGrad",
            "noneRed_smallimg-smoothGrad":"SmoothGrad",
            "noneRed_smallimg-guided":"GuidedBP",
            "interbyparts":"InterByParts",
            "abn":"ABN"}

def get_what_is_best():
    return {"Del":"min","Add":"max","DelCorr":"max","AddCorr":"max","IIC":"max","AD":"min","ADD":"max","Spars":"max","Time":"min"}

def get_metric_label():
    return {"Del":"DAUC","Add":"IAUC","Spars":"Sparsity","IIC":"IIC","AD":"AD","ADD":"ADD","DelCorr":"DC","AddCorr":"IC","Time":"Time","Acc":"Accuracy"}
    
def reverseLabDic(id_to_label,exp_id):

    label_to_id = {}

    for id in id_to_label:
        label = id_to_label[id]

        if label == "BR-NPA":
            if exp_id == "CUB10":
                id = "clus_masterClusRed"
            else:
                id = "clus_mast"
        elif id.startswith("noneRed"):
            id = "noneRed"

        label_to_id[label] = id 
    
    return label_to_id

def reformatAttScoreArray(attScores,pairs):

    k = attScores.shape[1]//pairs.shape[1]

    if k > 1:
        newAttScores = []
        for i in range(pairs.shape[0]):
            scoreList = []
            attScores_i = attScores[i].astype("float64")

            for j in range(pairs.shape[1]):
                scores,indices = torch.topk(torch.tensor(attScores_i),k=k)
                scoreList.append(scores.mean())
                attScores_i[indices] = -1

            newAttScores.append(scoreList) 
    else:
        newAttScores = attScores

    return np.array(newAttScores)

def correlation(points):
    corrList = []
    for i in range(len(points)):
        points_i = points[i]
        corrList.append(np.corrcoef(points_i,rowvar=False)[0,1])
    return np.array(corrList)

def attCorrelation(exp_id,img_bckgr):

    if not os.path.exists("../vis/{}/correlation/".format(exp_id)):
        os.makedirs("../vis/{}/correlation/".format(exp_id))

    suff = get_bckgr_suff(img_bckgr)

    for metric in ["Del","Add"]:
        csv_res = []
        csv_res_pop = []

        paths = getResPaths(exp_id,metric,img_bckgr)

        for path in paths:
            points = np.load(path,allow_pickle=True).astype("float")
            
            path_att_score = path.replace("attMetr{}{}".format(metric,suff),"attMetrAttScore")
            if os.path.exists(path_att_score):
                model_id = getModelId(path,metric,img_bckgr)   

                attScores = np.load(path_att_score,allow_pickle=True)
                oldShape = attScores.shape
                attScores = reformatAttScoreArray(attScores,points)
                if oldShape != attScores.shape:
                    np.save(path_att_score,attScores)

                points[:,:,0] = attScores

                points_diff = points.copy()
                if metric == "Del":
                    points_diff[:,:-1,1] = points[:,:-1,1]-points[:,1:,1]
                else:
                    points_diff[:,:-1,1] = points[:,1:,1]-points[:,:-1,1]
                points_diff = points_diff[:,:-1]

                all_corr = correlation(points_diff)

                corr = np.nanmean(all_corr)
  
                csv_res += "{},{}\n".format(model_id,corr) 
                csv_res_pop += "{},".format(model_id)+",".join([str(corr) for corr in all_corr])+"\n"

        with open(f"../results/{exp_id}/attMetrics_{metric}Corr{suff}.csv","w") as f:
            f.writelines(csv_res)

        with open(f"../results/{exp_id}/attMetrics_{metric}Corr{suff}_pop.csv","w") as f:
            f.writelines(csv_res_pop)

def normalize_metrics(values,metric):
    if metric in ["DelCorr","AddCorr"]:
        values = (values + 1)*0.5
    elif metric in ["AD","ADD","IIC"]:
        values = values*0.01 
    elif metric not in ["Del","Add"]:
        raise ValueError("Can't normalize",metric)
    
    return values 

def loadPerf(exp_id,metric,pop=True,img_bckgr=False,norm=False,reverse_met_to_min=False):

    suff = "-IB" if img_bckgr else ""

    if pop:
        perfs = np.genfromtxt("../results/{}/attMetrics_{}{}_pop.csv".format(exp_id,metric,suff),delimiter=",",dtype=str)
    else:
        if metric in ["ADD","AD","IIC"]:
            perfs = np.genfromtxt("../results/{}/attMetrics_Lift{}.csv".format(exp_id,suff),delimiter=",",dtype=str)
        
            if metric == "IIC":
                perfs = np.concatenate((perfs[:,0:1],perfs[:,1:2]),axis=1)
            elif metric == "AD":
                perfs = np.concatenate((perfs[:,0:1],perfs[:,2:3]),axis=1)
            elif metric == "ADD":
                perfs = np.concatenate((perfs[:,0:1],perfs[:,3:4]),axis=1) 
        else:
            perfs = np.genfromtxt("../results/{}/attMetrics_{}{}.csv".format(exp_id,metric,suff),delimiter=",",dtype=str)
    
    if norm:
        perfs_norm = normalize_metrics(perfs[:,1:].astype("float"),metric)
        perfs = np.concatenate((perfs[:,0:1],perfs_norm.astype(str)),axis=1)

    if get_what_is_best()[metric] == "min":
        perfs[:,1:]= (-1*perfs[:,1:].astype("float")).astype("str")

    return perfs

def kendallTauInd(metric_list,exp_id,img_bckgr,k,what_is_best,pop):
    rank_dic = {}
    kendall_tau_mat_k = np.zeros((len(metric_list),len(metric_list)))
    p_val_mat_k = np.zeros((len(metric_list),len(metric_list)))

    for i in range(len(metric_list)):

        metric = metric_list[i]

        if metric == "IIC":
            perfs = loadPerf(exp_id,metric,pop=False,img_bckgr=img_bckgr,reverse_met_to_min=True)
            rank = perfs[:,1].astype("float")
        else:
            perfs = loadPerf(exp_id,metric,pop=pop,img_bckgr=img_bckgr,reverse_met_to_min=True)
            rank = perfs[:,k+1].astype("float")
        rank_dic[metric] = rank 

    for i in range(len(metric_list)):
        for j in range(len(metric_list)):
            kendall_tau_mat_k[i,j],p_val_mat_k[i,j] = kendalltau(rank_dic[metric_list[i]],rank_dic[metric_list[j]])

    return kendall_tau_mat_k,p_val_mat_k

def computeKendallTauMat(metric_list,exp_id,pop,img_bckgr,what_is_best):
    
    kendall_tau_mat = np.zeros((len(metric_list),len(metric_list)))
    p_val_mat = np.zeros((len(metric_list),len(metric_list)))

    if pop:
        nbSamples = loadPerf(exp_id,"Del",pop=True,img_bckgr=img_bckgr).shape[0]

        for k in range(nbSamples):
            kendall_tau_mat_k,p_val_mat_k = kendallTauInd(metric_list,exp_id,img_bckgr,k,what_is_best,pop=True)
            kendall_tau_mat += kendall_tau_mat_k
            p_val_mat += p_val_mat_k

        kendall_tau_mat /= nbSamples
        p_val_mat /= nbSamples 

    else:
        kendall_tau_mat,p_val_mat = kendallTauInd(metric_list,exp_id,img_bckgr,0,what_is_best,pop=False)

    return kendall_tau_mat,p_val_mat

def ranking_similarities(exp_id,img_bckgr=False,pop=False):

    what_is_best = get_what_is_best()
    metric_list = ["Del","DelCorr","ADD","Add","AddCorr","IIC","AD"] 

    suff = "-IB" if img_bckgr else ""
    model_list = np.genfromtxt("../results/{}/attMetrics_Del{}.csv".format(exp_id,suff),delimiter=",",dtype=str)[:,0]

    kendall_tau_mat,p_val = computeKendallTauMat(metric_list,exp_id,pop,img_bckgr,what_is_best)

    cmap = plt.get_cmap("bwr")

    fig = plt.figure()
    ax = fig.gca()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    metric_to_label = get_metric_label()
    label_list = [metric_to_label[metric] for metric in metric_list]

    plt.imshow(kendall_tau_mat*0,cmap="Greys")
    #plt.imshow(p_val_mat*0)
    for i in range(len(metric_list)):
        for j in range(len(metric_list)):
            if i <= j:
                rad = 0.3
                circle = plt.Circle((i, j), rad, color=cmap((kendall_tau_mat[i,j]+1)*0.5))
                ax.add_patch(circle)

    fontSize = 17
    cbar = plt.colorbar(cm.ScalarMappable(norm=Normalize(-1,1),cmap=cmap))
    cbar.ax.tick_params(labelsize=fontSize)
    plt.xticks(range(len(metric_list)),label_list,rotation=45,fontsize=fontSize)
    plt.yticks(range(len(metric_list)),label_list,fontsize=fontSize)
    #fig, ax = plt.subplots()
    plt.tight_layout()
    suff += "_pop" if pop else ""
    print("../vis/{}/kendall_tau{}.png".format(exp_id,suff))
    plt.savefig("../vis/{}/kendall_tau{}.png".format(exp_id,suff))

def rankDist(x,y):
    tau = kendalltau(x,y)[0]
    if tau == -1:
        tau = -0.999

    return -np.log(0.5*(tau+1))

def run_dimred_or_load(path,allFeat,dimred="umap"):
    #if dimred == "pca":
    #    dimRedFunc = PCA
    #    kwargs = {}
    if dimred == "umap":
        dimRedFunc = umap.UMAP
        kwargs = {}
    elif dimred == "tsne":
        dimRedFunc = TSNE
        kwargs = {"metric":rankDist,"learning_rate":100,"init":"pca"}
    else:
        raise ValueError("Unknown dimred {}".format(dimred))

    path = path.replace(".npy","_"+dimred+".npy")

    if os.path.exists(path):
        allFeat_umap = np.load(path,allow_pickle=True)
        if len(allFeat_umap.shape) == 0:
            allFeat_umap = allFeat_umap.item()
    else:
        if type(allFeat) is dict:
            allFeat_umap = {}
        else:
            allFeat_umap = None

    if type(allFeat) is dict:

        for metric in allFeat:
            if not metric in allFeat_umap:
                allFeat_umap[metric] = {}

        for metric in allFeat:
            for bckgr in allFeat[metric]:
                if not bckgr in allFeat_umap[metric]:
                    np.random.seed(0)
                    allFeat[metric][bckgr] = dimRedFunc(n_components=2,**kwargs).fit_transform(allFeat[metric][bckgr])
                else:
                    allFeat[metric][bckgr] = allFeat_umap[metric][bckgr]
                    
    else:
        allFeat = dimRedFunc(n_components=2,**kwargs).fit_transform(allFeat) if allFeat_umap is None else allFeat_umap

    np.save(path,allFeat)

    return allFeat 

def dimred_metrics(exp_id,pop=False,dimred="umap",img_bckgr=False):

    metric_list = ["Del","DelCorr","ADD","Add","AddCorr","AD","IIC"] 
    if pop:
        metric_list.pop(-1)

    cmap = plt.get_cmap("Set1")
    colorInds = [0,4,5,1,2,3]
    colors = [cmap(colorInds[i]*1.0/8) for i in range(len(metric_list))]

    allPerfs = []
    sample_nb = loadPerf(exp_id,"Del",pop=pop).shape[1]-1
    for metric in metric_list:
        if pop:
                perfs = loadPerf(exp_id,metric,pop=True,img_bckgr=img_bckgr,norm=True,reverse_met_to_min=True)

                perfs = perfs[:,1:].transpose(1,0)
                allPerfs.append(perfs)
        else:
            perfs = loadPerf(exp_id,metric,pop=False,img_bckgr=img_bckgr,norm=True,reverse_met_to_min=True)
            allPerfs.append(perfs[np.newaxis,:,1])

    allPerfs = np.concatenate(allPerfs,axis=0).astype("float")

    path = f"../results/{exp_id}/metrics_dimred_pop{pop}_imgBckr{img_bckgr}.npy"
    allFeat = run_dimred_or_load(path,allPerfs,dimred)  
    
    metric_to_label = get_metric_label()
    plt.figure()
    colorList = []
    for i,metric in enumerate(metric_list):

        start,end = i*sample_nb,(i+1)*sample_nb
        plt.scatter([allFeat[start,0]],[allFeat[start,1]],label=metric_to_label[metric],color=colors[i])
        colorList.extend([colors[i] for _ in range(sample_nb)])

    print(allFeat.shape,np.array(colorList).shape)
    feat_and_color = np.concatenate((allFeat,np.array(colorList)),axis=1)
    np.random.shuffle(feat_and_color)

    allFeat,colorList = feat_and_color[:,:2],feat_and_color[:,2:]

    fontSize = 15
    plt.xticks(fontsize=fontSize)
    plt.yticks(fontsize=fontSize)
    plt.scatter([allFeat[:,0]],[allFeat[:,1]],color=colorList)
    plt.legend(fontsize=fontSize-2)
    plt.savefig(f"../vis/{exp_id}/metrics_{dimred}_pop{pop}_imgBckgr{img_bckgr}.png")
    plt.close()

def get_sec_model_list():
    return ["SVM","KNN","DT","NN"]

def get_metric_list(full=False):
    if full:
        return ["Del","Add","Lift","DelCorr","AddCorr"]
    else:
        return ["Del","Add","Lift"]

def make_feature_path(varying_variable,exp_id,metric,value,model_id):
    if varying_variable in ["resolution","sparsity"]:
        featPath = f"../results/{exp_id}/attMetrFeat{metric}_{model_id}{value}.npy"
    else:
        featPath = f"../results/{exp_id}/attMetrFeat{metric}{value}_{model_id}.npy"
    return featPath

def get_resolution_list(exp_id):
    paths = glob.glob(f"../results/{exp_id}/attMetrFeatDel_noneRed-gradcam_pp-res*.npy")
    resolution_list = sorted(list(map(lambda x:int(os.path.basename(x).split("-res")[1].replace(".npy","")),paths)))
    value_list = list(map(lambda x:"-res"+str(x),resolution_list))
    return value_list 

def get_paths_and_info(varying_variable,exp_id,file_paths,value_list=None):

    if file_paths[0].find("attMetrFeat") != -1:
        res_suff = "-res" 
        spar_suff = "-spar"
    elif file_paths[0].find("attMetrReprSep") != -1:
        res_suff = "resolution" 
        spar_suff = "sparsity"
    else:
        raise ValueError("Unknown file format",file_paths[:10])

    value_list_dic = {"background":get_backgr_list(),"sparsity":get_spars_list(),"resolution":get_resolution_list(exp_id)}
    suff_dic = {"background":"","sparsity":spar_suff,"resolution":res_suff}
    
    if value_list is None:
        value_list = value_list_dic[varying_variable]
    elif varying_variable in ["sparsity","resolution"]:
        value_list = list(map(lambda x:suff_dic[varying_variable]+x,value_list))

    if varying_variable == "background":
        label = "bckgr"
        filename_suff = ""
        file_paths = list(filter(lambda x:x.find(spar_suff) == -1,file_paths))
        file_paths = list(filter(lambda x:x.find(res_suff) == -1,file_paths))
    elif varying_variable == "sparsity":
        label = varying_variable
        filename_suff = "_"+varying_variable
        file_paths = list(filter(lambda x:x.find(spar_suff) != -1,file_paths))
    else:
        label = varying_variable
        filename_suff = "_"+varying_variable
        file_paths = list(filter(lambda x:x.find(res_suff) != -1,file_paths))
    return label,filename_suff,value_list,file_paths

def ood_repr(exp_id,quantitative,varying_variable,metric_list=None,value_list=None):

    file_paths = glob.glob(f"../results/{exp_id}/attMetrFeat*_*.npy")

    label,filename_suff,value_list,file_paths = get_paths_and_info(varying_variable,exp_id,file_paths,value_list)

    if metric_list is None:
        metric_list = get_metric_list()

    os.makedirs(f"../vis/{exp_id}/representation_study/",exist_ok=True)

    path_to_model_id = lambda x:"_".join(x.split("/")[-1].split("_")[1:]).replace(".npy","").split("-res")[0].split("-spar")[0]
    model_ids = set(list(map(path_to_model_id,file_paths)))
    model_ids = list(filter(lambda x:x.find("umap") == -1,model_ids))

    for model_id in model_ids:
        print(model_id)
        sample_nb_dic = {}
        step_nb_dic = {}
        allFeat = {}
        for metric in metric_list:
            allFeat[metric] = {}
            sample_nb_dic[metric] = {}
            step_nb_dic[metric] = {}
            for value in value_list:
                featPath = make_feature_path(varying_variable,exp_id,metric,value,model_id)
                if os.path.exists(featPath):
                    feat = np.load(featPath,mmap_mode="r")
                    sample_nb_dic[metric][value],step_nb_dic[metric][value] = feat.shape[0],feat.shape[1]
                    allFeat[metric][value] = feat.reshape(feat.shape[0]*feat.shape[1],-1)

        umap_path = f"../results/{exp_id}/attMetrFeat_{model_id}{filename_suff}.npy"
        if not quantitative:
            allFeat = run_dimred_or_load(umap_path,allFeat,dimred="umap")

        sec_model_name_list = get_sec_model_list()

        #We will train models to separate representations of altered images from representations of regular images
        constDic = {"SVM":svm.SVC,"DT":tree.DecisionTreeClassifier,"KNN":neighbors.KNeighborsClassifier,
                    "NN":neural_network.MLPClassifier}

        kwargsDic = {"NN":{"hidden_layer_sizes":(128,),"batch_size":135 if metric=="Lift" else 2000,
                           "early_stopping":True,"learning_rate":"adaptive"},
                     "SVM":{"probability":True},"DT":{},"KNN":{}}

        metrDic = {"Del":(plt.get_cmap("plasma"),lambda x:""),\
                   "Add":(lambda x:plt.get_cmap("plasma")(1-x),lambda x:""),
                   "Lift":(lambda x:[plt.get_cmap("plasma")(0),"red","green"][int(x*step_nb_dic["Lift"][value_list[0]])],lambda x:["Original Data","Mask","Inversed mask"][x])}

        for metric in metric_list:
            print("\t",metric)
            cmap,labels = metrDic[metric]

            for value in value_list:
                print("\t\t",value)

                step_nb_met,sample_nb = step_nb_dic[metric][value],sample_nb_dic[metric][value]

                if value in allFeat[metric]:
                    feat = allFeat[metric][value].copy()
                    feat = feat.reshape(sample_nb,step_nb_met,-1)

                    if quantitative:
                        np.random.seed(0) 
                        np.random.shuffle(feat)
                        train_x = feat[:int(len(feat)*0.5)]
                        val_x = feat[int(len(feat)*0.5):]

                        labels = np.arange(feat.shape[1])
                        if metric in ["Lift","Del"]:
                            labels = 1.0*(labels > 0)
                        else:
                            labels = 1.0*(labels < feat.shape[1] - 1)

                        train_y = np.repeat(labels[np.newaxis],len(train_x),0).reshape(-1)
                        val_y = np.repeat(labels[np.newaxis],len(val_x),0).reshape(-1)
             
                        train_x = train_x.reshape(-1,train_x.shape[-1])
                        val_x = val_x.reshape(-1,val_x.shape[-1])
           
                        perfCSVPath = f"../results/{exp_id}/attMetrReprSep{metric}_{model_id}{filename_suff}.csv"
                        if not os.path.exists(perfCSVPath):
                            with open(perfCSVPath,"w") as file:
                                print(f"{label},sec_model,train_acc,train_auc,val_acc,val_auc",file=file) 

                        for secModel in sec_model_name_list:
        
                            arr = np.genfromtxt(perfCSVPath,delimiter=",",dtype=str)
                            
                            if len(arr.shape) > 1:
                                inds = np.multiply(arr[:,0]==value,arr[:,1]==secModel)
                                if len(inds) > 0:
                                    modelAlreadyTrained = len(np.argwhere(inds)[:,0]) > 0
                                else:
                                    modelAlreadyTrained = False 
                            #No model has been trained already 
                            #Or this model has not been already 
                            if len(arr.shape) == 1 or not modelAlreadyTrained: 
                                print(f"\t\t\tModel {secModel} being trained")  
                                np.random.seed(0)              
                                model = constDic[secModel](**kwargsDic[secModel])
                                model.fit(train_x,train_y)

                                if model_id == "noneRed-gradcam_pp" and metric == "Del" and \
                                    ((value == "-res14" and varying_variable =="resolution") or \
                                    (value == "-spar1.0" and varying_variable =="sparsity") or \
                                    (value == "" and varying_variable =="background")):

                                    np.save(f"../results/EMB10/train_x_{varying_variable}.npy",train_x)
                                    np.save(f"../results/EMB10/train_y_{varying_variable}.npy",train_y)
                                    np.save(f"../results/EMB10/val_x_{varying_variable}.npy",val_x)
                                    np.save(f"../results/EMB10/val_y_{varying_variable}.npy",val_y) 
                                    np.save(f"../results/EMB10/model_{varying_variable}.npy",model)                                 

                                train_y_score = model.predict_proba(train_x)[:,1]
                                train_acc = model.score(train_x,train_y)
                                train_auc = sklearn.metrics.roc_auc_score(train_y,train_y_score)
                                
                                val_y_score = model.predict_proba(val_x)[:,1]
                                val_acc = model.score(val_x,val_y)            
                                val_auc = sklearn.metrics.roc_auc_score(val_y,val_y_score)

                                with open(perfCSVPath,"a") as file:
                                    print(f"{value},{secModel},{train_acc},{train_auc},{val_acc},{val_auc}",file=file)

                            else:
                                print(f"\t\t\tModel {secModel} already trained.")

                    else:
                        plt.figure()
                        pts_inds = np.arange(feat.shape[1])
                        np.random.shuffle(pts_inds)

                        for i in pts_inds:
                            feat_i = feat[:,i]
                            plt.scatter(feat_i[:,0],feat_i[:,1],marker="o",color=cmap(i*1.0/feat.shape[1]),alpha=0.5,label=labels(i))
                        
                        plt.xlim(feat[:,:,0].min()-1,feat[:,:,0].max()+1)
                        plt.ylim(feat[:,:,1].min()-1,feat[:,:,1].max()+1)
                        
                        if metric in ["Add","Del"]:
                            plt.colorbar(cm.ScalarMappable(norm=Normalize(0,1),cmap=plt.get_cmap("plasma")))
                        else:
                            plt.legend()

                        figPath = f"../vis/{exp_id}/representation_study/attMetrFeatDimRed{metric}{value}_{model_id}{filename_suff}.png"

                        plt.savefig(figPath)
                        plt.close()

def repr_path_to_model_id(path):
    return "_".join(path.split("/")[-1].split("ReprSep")[1].split("_")[1:]).replace(".csv","")

def imshow_perf_matrix(perf_matrix,fig_path,xlabels,ylabels):
    plt.figure()
    cmap_name = "plasma"
    cmap = plt.get_cmap(cmap_name)
    plt.imshow(perf_matrix,cmap=cmap_name,vmin=0,vmax=100)
    plt.yticks(np.arange(len(ylabels)),ylabels)
    plt.xticks(np.arange(len(xlabels)),xlabels)
    for i in range(len(perf_matrix)):
        for j in range(len(perf_matrix[i])):
            color = "black" if perf_matrix[i,j] > 0.5 else "white"
            plt.text(j-.1,i,round(perf_matrix[i,j],1),color=color)
            
    plt.colorbar(cm.ScalarMappable(norm=Normalize(0,1),cmap=cmap))
    plt.tight_layout()
    plt.savefig(fig_path)
    plt.close()

def get_backgr_list():
    return ["-IB","","-white","-gray","-blur","-highpass","-lowpass"]

def get_spars_list():
    spar_values =  np.array([0.06, 0.125, 0.25, 0.5, 1, 2, 4, 8]).astype("str")
    spar_values = list(map(lambda x:"-spar"+x,spar_values))
    return spar_values

def get_backgr_label_dic():
    return {"-IB":"IB","":"Black","-white":"White","-gray":"Gray","-blur":"Blur",\
            "-highpass":"High pass","-lowpass":"Low Pass"}

def viz_ood_repr(exp_id,varying_variable):

    paths = glob.glob(f"../results/{exp_id}/attMetrReprSep*_*.csv")
    _,filename_suff,value_list,paths = get_paths_and_info(varying_variable,exp_id,paths)

    if varying_variable == "background":
        labels = get_backgr_label_dic()
    elif varying_variable == "sparsity":
        labels = {spar:float(spar.replace("-spar","")) for spar in value_list}
    else:
        labels = {res:float(res.replace("-res","")) for res in value_list}

    os.makedirs(f"../vis/{exp_id}/representation_study/val_auc/",exist_ok=True)
    os.makedirs(f"../vis/{exp_id}/representation_study/val_acc/",exist_ok=True)

    for path in paths:
        metric = path.split("/")[-1].split("ReprSep")[1].split("_")[0]
        model_id = repr_path_to_model_id(path)

        csv = np.genfromtxt(path,delimiter=",",dtype=str)

        sec_model_list = get_sec_model_list()
    
        for i in range(2):
            sec_model_metric_ind = csv.shape[1]-2+i
            sec_model_metric = csv[0,sec_model_metric_ind]

            perf_matrix = []
            for value in value_list:
                csv_value = csv[1:][csv[1:,0] == value]
    
                row = []
                for sec_model in sec_model_list:
                    csv_sec_model = csv_value[(csv_value[:,1] == sec_model)]
                    csv_sec_model_metric = csv_sec_model[:,sec_model_metric_ind]
                    if len(csv_sec_model_metric) == 0:
                        value = 0
                    else:
                        value = float(csv_sec_model_metric[0])

                    row.append(value)

                perf_matrix.append(row)

            
            fig_path = f"../vis/{exp_id}/representation_study/{sec_model_metric}/attMetrFeat{metric}_{model_id}_{sec_model_metric}{filename_suff}.png"
            print(fig_path)
            valueLabelList = [labels[value] for value in value_list]
            perf_matrix = np.array(perf_matrix)
            imshow_perf_matrix(100*perf_matrix,fig_path,sec_model_list,valueLabelList)

def agr_ood_repr(exp_id):

    metric_list = get_metric_list()
    backgr_list = get_backgr_list()
    sec_model_list = get_sec_model_list()

    os.makedirs(f"../results/{exp_id}/representation_study/",exist_ok=True)

    for metric in metric_list:

        perfCSVPaths = glob.glob(f"../results/{exp_id}/attMetrReprSep{metric}_*.csv")

        perfDic = {}

        for path in perfCSVPaths:

            csv = np.genfromtxt(path,delimiter=",",dtype=str)

            bckgr_col = np.argwhere(csv[0,:]=="bckgr")[0,0]
            sec_model_col = np.argwhere(csv[0,:]=="sec_model")[0,0]
            val_auc_col = np.argwhere(csv[0,:]=="val_auc")[0,0]

            for row in csv[1:]:
                if row[bckgr_col] not in perfDic:
                    perfDic[row[bckgr_col]] = {}
                
                if row[sec_model_col] not in perfDic[row[bckgr_col] ]:
                    perfDic[row[bckgr_col] ][row[sec_model_col]] = []

                perfDic[row[bckgr_col]][row[sec_model_col]].append(float(row[val_auc_col]))

        backgr_label_dic = get_backgr_label_dic()
        backgr_label_list = [backgr_label_dic[bckgr] for bckgr in backgr_list]

        perfMat = []  
        csv = "&"+"&".join(sec_model_list)+"\\\\ \n"     

        for bckgr in backgr_list:
            row = []
            for sec_model in sec_model_list:
                mean_value = sum(perfDic[bckgr][sec_model])
                mean_value /= len(perfDic[bckgr][sec_model])
                row.append(str(round(100*mean_value,1))) 
            perfMat.append(row)
            csv += backgr_label_dic[bckgr]+"&"+"&".join(row)+"\\\\ \n"

        table_path = f"../results/{exp_id}/representation_study/attMetrFeat{metric}.tex"

        with open(table_path,"w") as file:
            print(csv,file=file,end="") 

        fig_path = table_path.replace("results","vis").replace(".tex",".png")
        perfMat = np.array(perfMat).astype("float")

        #Reordering rows to show the best bckgr on top
        allRankings = []
        for j,sec_model in enumerate(sec_model_list):
            print(perfMat[:,j].argsort())
            argsort = np.argsort(-perfMat[:,j].astype("float"))
        
            ranking = np.zeros(len(argsort))

            for i in range(len(ranking)):
                ranking[argsort[i]] = i

            allRankings.append(ranking)

        average_ranking = np.stack(allRankings,0)
        average_ranking = average_ranking.mean(axis=0)

        perfMat = perfMat[average_ranking.argsort()]
        backgr_label_list = np.array(backgr_label_list)[average_ranking.argsort()]
        imshow_perf_matrix(perfMat,fig_path,sec_model_list,backgr_label_list)

def vote_for_best_model(exp_id,metric_list,img_bckgr):

    allRankings = []
    for metric in metric_list:
        perfs = loadPerf(exp_id,metric,img_bckgr=img_bckgr,reverse_met_to_min=True,pop=False)
        argsort = np.argsort(-perfs[:,1].astype("float"))
        
        ranking = np.zeros(len(argsort))

        for i in range(len(ranking)):
            ranking[argsort[i]] = i+1

        allRankings.append(ranking)

    average_ranking = np.stack(allRankings,0)
    average_ranking = average_ranking.mean(axis=0)

    return average_ranking

def find_best_methods(exp_id,img_bckgr):

    model_list = loadPerf(exp_id,"Del")[:,0]

    ranking_del = vote_for_best_model(exp_id,["Del","DelCorr","ADD"],img_bckgr)
    ranking_add = vote_for_best_model(exp_id,["Add","AddCorr","AD","IIC"],img_bckgr)
    ranking_all = vote_for_best_model(exp_id,["Del","DelCorr","ADD","Add","AddCorr","AD","IIC"],img_bckgr)

    rank_list = [ranking_del,ranking_add,ranking_all]
    rank_label = ["Deletion","Addition","Global"]

    id_to_label = get_id_to_label()

    for i in range(len(rank_list)):

        ranking = rank_list[i]

        print(rank_label[i])
        argsort = np.argsort(ranking)

        for ind in argsort:
            print("\t",round(ranking[ind],2),id_to_label[model_list[ind]])

    print(model_list[np.argmin(ranking_del)],model_list[np.argmin(ranking_add)],model_list[np.argmin(ranking)])

def viz_resolution(exp_id,img_bckgr):

    model_id_pref = "noneRed-res"
    metric_list = get_metric_list(full=True)

    metric_list.remove("Lift")
    metric_list.extend(["AD","ADD","IIC"])

    bckgr_suff = get_bckgr_suff(img_bckgr)
    metric_to_label= get_metric_label()

    for metric in metric_list:

        if metric == "IIC":
            suff = ""
            csv = pd.read_csv(f"../results/{exp_id}/attMetrics_Lift{bckgr_suff}{suff}.csv",header=None)
        else:
            suff = "_pop"
            csv = pd.read_csv(f"../results/{exp_id}/attMetrics_{metric}{bckgr_suff}{suff}.csv",header=None)
 
        csv = csv.loc[csv[0].str.contains(model_id_pref)]
        resolution = csv.apply(lambda x:int(x[0].split("-res")[1]),axis=1)
        csv["resolution"] = resolution
        csv = csv.sort_values(by="resolution")

        if metric == "IIC":
            submetrics = np.array(get_lift_submetric())
            col_ind = np.argwhere(submetrics=="IIC").item()+1
            plot_metric(csv,col_ind,metric_to_label,"IIC",exp_id) 
        else:
            plot_metric(csv,np.arange(len(csv.columns))[1:-1],metric_to_label,metric,exp_id)

def plot_metric(csv,col_inds,metric_to_label,metric,exp_id):
    
    plt.figure()
    if type(col_inds) is np.ndarray:
        plt.violinplot(np.array(csv[col_inds]).transpose(1,0))
        plt.xticks(np.arange(len(csv["resolution"]))+1,csv["resolution"].astype("str"))
    else:
        plt.plot(list(csv[col_inds]),"*-")
        plt.xticks(np.arange(len(csv["resolution"])),csv["resolution"].astype("str")) 
    plt.xlabel("Resolution")
    plt.ylabel(metric_to_label[metric])        
    plt.savefig(f"../vis/{exp_id}/resolution_{metric_to_label[metric]}.png")
    plt.close()

def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--att_metrics',action="store_true") 
    argreader.parser.add_argument('--all_att_metrics',action="store_true") 
    argreader.parser.add_argument('--with_std',action="store_true") 
    argreader.parser.add_argument('--img_bckgr',type=str,default="black")
    argreader.parser.add_argument('--resolution',type=int)
    argreader.parser.add_argument('--pop',action="store_true")   
    argreader.parser.add_argument('--ood_repr',action="store_true") 
    argreader.parser.add_argument('--quantitative',action="store_true") 
    argreader.parser.add_argument('--varying_variable',type=str) 
    argreader.parser.add_argument('--viz_ood_repr',action="store_true")
    argreader.parser.add_argument('--agr_ood_repr',action="store_true")
    argreader.parser.add_argument('--dimred_metrics',action="store_true") 
    argreader.parser.add_argument('--dimred_func',type=str,default="tsne") 
    argreader.parser.add_argument('--ranking_similarities',action="store_true") 
    argreader.parser.add_argument('--find_best_methods',action="store_true") 
    argreader.parser.add_argument('--metrics',type=str,nargs="*") 
    argreader.parser.add_argument('--value_list',type=str,nargs="*") 
    argreader.parser.add_argument('--viz_resolution',action="store_true") 

    ###################################### Accuracy per video ############################################""

    argreader.parser.add_argument('--accuracy_per_video',action="store_true") 
    
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
    argreader.parser.add_argument('--ind_to_keep',type=int,nargs="*",metavar="INT",help='The index of the images to keep')
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

    if args.att_metrics:
        for metric in ["Add","Del","Spars","Lift"]:
            attMetrics(args.exp_id,metric,args.img_bckgr)
        
        attCorrelation(args.exp_id,img_bckgr=args.img_bckgr)
    if args.viz_resolution:
        viz_resolution(args.exp_id,args.img_bckgr)
    if args.ood_repr:
        ood_repr(args.exp_id,args.quantitative,args.varying_variable,args.metrics,args.value_list)
    if args.viz_ood_repr:
        viz_ood_repr(args.exp_id,args.varying_variable)
    if args.agr_ood_repr:
        agr_ood_repr(args.exp_id)
    if args.dimred_metrics:
        dimred_metrics(args.exp_id,args.pop,args.dimred_func,args.img_bckgr)
    if args.ranking_similarities:
        ranking_similarities(args.exp_id,img_bckgr=args.img_bckgr,pop=args.pop)
    if args.find_best_methods:
        find_best_methods(args.exp_id,args.img_bckgr)
    if args.show_maps:

        #Setting default values
        default_values = {"pond_by_norm":True,"only_norm":False,"interp":False,"direct_ind":False,"maps_inds":-1,
                          "sparsity_factor":1}
        for key in default_values:
            param = getattr(args,key)
            if len(param) == 0:
                param = [default_values[key] for _ in range(len(args.model_ids))]
            setattr(args,key,param)

        showSalMaps(args.exp_id,args.img_nb,args.plot_id,args.nrows,args.class_index,args.ind_to_keep,args.viz_id,args,
                    args.model_ids,args.expl,args.maps_inds,args.pond_by_norm,args.only_norm,args.interp,args.direct_ind,
                    args.sparsity_factor)



if __name__ == "__main__":
    main()
