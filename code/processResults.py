
from args import ArgReader
import os
import glob

import torch
import numpy as np

import matplotlib.pyplot as plt
from matplotlib import cm 
plt.switch_backend('agg')

from sklearn.manifold import TSNE
import sklearn
from sklearn import svm ,neural_network,tree,ensemble,neighbors,gaussian_process
import matplotlib
import matplotlib.cm as cm

import load_data
import scipy
import torch.nn.functional as F

import umap
from sklearn.manifold import  TSNE

import scipy.stats

from scipy.signal import savgol_filter
from scipy.stats import kendalltau

import sys

def get_metrics_with_IB():
    return ["Del","DelCorr","Add","AddCorr","AD","ADD","IIC","Lift"]

def getResPaths(exp_id,metric,img_bckgr):
    if img_bckgr and metric in get_metrics_with_IB():
        paths = sorted(glob.glob("../results/{}/attMetr{}-IB_*.npy".format(exp_id,metric)))
    else:
        paths = sorted(glob.glob("../results/{}/attMetr{}_*.npy".format(exp_id,metric)))
        paths = list(filter(lambda x:os.path.basename(x).find("-IB") ==-1,paths))

    paths = removeOldFiles(paths)

    return paths 

def removeOldFiles(paths):
    paths = list(filter(lambda x:os.path.basename(x).find("noise") ==-1,paths))
    paths = list(filter(lambda x:os.path.basename(x).find("imgBG") ==-1,paths))
    paths = list(filter(lambda x:os.path.basename(x).find("maskFeat") ==-1,paths))
    return paths

def getModelId(path,metric,img_bckgr):
    suff = "-IB" if img_bckgr else ""
    model_id = os.path.basename(path).split("attMetr{}{}_".format(metric,suff))[1].split(".npy")[0]
    return model_id

def attMetrics(exp_id,metric="Del",ignore_model=False,img_bckgr=False):

    suff = metric

    paths = getResPaths(exp_id,metric,img_bckgr)

    resDic = {}
    resDic_pop = {}

    if ignore_model:
        _,modelToIgn = getIndsToUse(paths,metric)
    else:
        modelToIgn = []

    if metric in ["Del","Add"]:
        for path in paths:

            model_id = getModelId(path,metric,img_bckgr)
            
            if model_id not in modelToIgn:
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
            
            if model_id not in modelToIgn:
                
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
            
            path = path.replace("-IB","")
            model_id = getModelId(path,metric,img_bckgr=False)
            
            if model_id not in modelToIgn:
                sparsity_list = 1/np.load(path,allow_pickle=True)

            resDic_pop[model_id] = np.array(sparsity_list)
            resDic[model_id] = resDic_pop[model_id].mean() 

    suff += "-IB" if img_bckgr and metric != "Spars" else ""
    csv = "\n".join(["{},{}".format(key,resDic[key]) for key in resDic])
    with open("../results/{}/attMetrics_{}.csv".format(exp_id,suff),"w") as file:
        print(csv,file=file)

    if metric == "Lift":
        for metric in ["IIC","AD","ADD"]:
            csv = "\n".join(["{},{}".format(key,",".join(resDic_pop[key][metric].astype("str"))) for key in resDic_pop])
            suff = "-IB" if img_bckgr else ""
            with open("../results/{}/attMetrics_{}{}_pop.csv".format(exp_id,metric,suff),"w") as file:
                print(csv,file=file)

    else:
        csv = "\n".join(["{},{}".format(key,",".join(resDic_pop[key].astype("str"))) for key in resDic_pop])
        with open("../results/{}/attMetrics_{}_pop.csv".format(exp_id,suff),"w") as file:
            print(csv,file=file)

def getIndsToUse(paths,metric):
    
    modelToIgn = []

    model_targ_ind = 0

    while model_targ_ind < len(paths) and not os.path.exists(paths[model_targ_ind].replace("Add","Targ").replace("Del","Targ")):
        model_targ_ind += 1

    if model_targ_ind == len(paths):
        use_all_inds = True
    else:
        use_all_inds = False 
        targs = np.load(paths[model_targ_ind],allow_pickle=True)
        
        indsToUseBool = np.array([True for _ in range(len(targs))])
        indsToUseDic = {}

    for path in paths:
        
        model_id = os.path.basename(path).split("attMetr{}_".format(metric))[1].split(".npy")[0]
        
        model_id_nosuff = model_id.replace("-max","").replace("-onlyfirst","").replace("-fewsteps","")

        predPath = path.replace(metric,"Preds").replace(model_id,model_id_nosuff)

        if not os.path.exists(predPath):
            predPath = path.replace(metric,"PredsAdd").replace(model_id,model_id_nosuff)

        if os.path.exists(predPath) and not use_all_inds:
            preds = np.load(predPath,allow_pickle=True)

            if preds.shape != targs.shape:
                inds = []
                for i in range(len(preds)):
                    if i % 2 == 0:
                        inds.append(i)

                preds = preds[inds]
            
            indsToUseDic[model_id] = np.argwhere(preds==targs)
            indsToUseBool = indsToUseBool*(preds==targs)
  
        else:
            modelToIgn.append(model_id)
            print("no predpath",predPath)

    if use_all_inds:
        indsToUse = None
    else:
        indsToUse =  np.argwhere(indsToUseBool)
        
    return indsToUse,modelToIgn 

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

def get_label_to_id():
    id_to_label = get_id_to_label()
    return {id_to_label[id]:id for id in id_to_label}

def get_is_post_hoc():
    return {"bilRed":False,
            "bilRed_1map":False,
            "clus_masterClusRed":False,
            "clus_mast":False,
            "noneRed":True,
            "protopnet":False,
            "prototree":False,
            "noneRed-gradcam":True,
            "noneRed-gradcam_pp":True,
            "noneRed-score_map":True,
            "noneRed-ablation_cam":True,
            "noneRed-rise":True,
            "noneRed_smallimg-varGrad":True,
            "noneRed_smallimg-smoothGrad":True,
            "noneRed_smallimg-guided":True,
            "interbyparts":False,
            "abn":False}

def ttest_attMetr(exp_id,metric="del",img_bckgr=False):

    id_to_label = get_id_to_label()

    suff = metric
    suff += "-IB" if img_bckgr and metric in get_metrics_with_IB() else ""

    arr = np.genfromtxt("../results/{}/attMetrics_{}_pop.csv".format(exp_id,suff),dtype=str,delimiter=",")

    metric_to_max = []
    what_is_best = get_what_is_best()
    for metric_ in what_is_best:
        if what_is_best[metric_] == "max":
            metric_to_max.append(metric_)

    arr = best_to_worst(arr,ascending=metric in metric_to_max)

    model_ids = arr[:,0]

    labels = [id_to_label[model_id] for model_id in model_ids]

    res_mat = arr[:,1:].astype("float")

    if metric == "add":
        rnd_nb = 2
    elif metric == "del":
        rnd_nb = 3 
    else:
        rnd_nb = 1

    perfs = [(str(round(mean,rnd_nb)),str(round(std,rnd_nb))) for (mean,std) in zip(res_mat.mean(axis=1),res_mat.std(axis=1))]

    p_val_mat = np.zeros((len(res_mat),len(res_mat)))
    for i in range(len(res_mat)):
        for j in range(len(res_mat)):
            p_val_mat[i,j] = scipy.stats.ttest_ind(res_mat[i],res_mat[j],equal_var=False)[1]

    p_val_mat = (p_val_mat<0.05)

    res_mat_mean = res_mat.mean(axis=1)

    diff_mat = np.abs(res_mat_mean[np.newaxis]-res_mat_mean[:,np.newaxis])
    
    diff_mat_norm = (diff_mat-diff_mat.min())/(diff_mat.max()-diff_mat.min())

    cmap = plt.get_cmap('plasma')

    fig = plt.figure()

    ax = fig.gca()

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    plt.imshow(p_val_mat*0,cmap="Greys")
    for i in range(len(res_mat)):
        for j in range(len(res_mat)):
            if i <= j:
                rad = 0.3 if p_val_mat[i,j] else 0.1
                circle = plt.Circle((i, j), rad, color=cmap(diff_mat_norm[i,j]))
                ax.add_patch(circle)

    plt.yticks(np.arange(len(res_mat)),labels)
    plt.xticks(np.arange(len(res_mat)),["" for _ in range(len(res_mat))])
    plt.colorbar(cm.ScalarMappable(norm=matplotlib.colors.Normalize(diff_mat.min(),diff_mat.max()),cmap=cmap))
    for i in range(len(res_mat)):
        plt.text(i-0.2,i-0.4,labels[i],rotation=45,ha="left")
    plt.tight_layout()
    plt.savefig("../vis/{}/ttest_{}_attmetr.png".format(exp_id,suff))

def best_to_worst(arr,ascending=True):

    if not ascending:
        key = lambda x:-x[1:].astype("float").mean()
    else:
        key = lambda x:x[1:].astype("float").mean()

    arr = np.array(sorted(arr,key=key))

    return arr

def loadArr(exp_id,metric,best,img_bckgr):
    
    suff = "-IB" if img_bckgr and metric in get_metrics_with_IB() else ""
    arr = np.genfromtxt("../results/{}/attMetrics_{}{}_pop.csv".format(exp_id,metric,suff),dtype=str,delimiter=",") 
    arr_f = arr[:,1:].astype("float")
    if best == "max":
        best = arr_f.mean(axis=1).max()
    else:
        best = arr_f.mean(axis=1).min()
    return arr,arr_f,best

def find_perf(i,metric,arr_dic,arr_f_dic,best_dic):
    ind = np.argwhere(arr_dic[metric][:,0] == arr_dic["Del"][i,0])[0,0]
    mean = arr_f_dic[metric][ind].mean()

    if metric in ["Add","DelCorr","AddCorr"]:
        mean_rnd,std_rnd = round(mean,3),round(arr_f_dic[metric][ind].std(),3)
    elif metric in ["IIC"]:
        mean_rnd,std_rnd = round(mean*0.01,3),0
    elif metric in ["AD","ADD"]:
        mean_rnd,std_rnd = round(mean*0.01,3),round(arr_f_dic[metric][ind].std()*0.01,3)
    else:
        raise ValueError("Unkown metric",metric)

    is_best = (mean==best_dic[metric])
    return mean_rnd,std_rnd,is_best

def processRow(csv,row,with_std,metric_list,full=True,postHoc=False,postHocInd=None,nbPostHoc=None):
    
    for metric in metric_list:

        if metric != "Acc" or not postHoc:
            csv += latex_fmt(row[metric],row[metric+"_std"],row["is_best_"+metric] == "True",\
                        with_std=with_std and metric != "IIC",with_start_char=(metric!="Accuracy"))
        else:
            if full:
                if postHocInd > 0:
                    csv += "&"
                else:
                    acc_text = latex_fmt(row["Acc"],row["Acc_std"],row["is_best_Acc"] == "True",with_std=with_std and metric != "IIC",with_start_char=False)
                    csv += "&\multirow{"+str(nbPostHoc)+"}{*}{$"+acc_text+"$}"

    csv += "\\\\ \n"
    return csv 

def formatRow(res_list,i):
    row = res_list[i]
    row = {key:str(row[key]) for key in row}
    return row

def get_what_is_best():
    return {"Del":"min","Add":"max","DelCorr":"max","AddCorr":"max","IIC":"max","AD":"min","ADD":"max","Spars":"max","Time":"min"}

def get_metric_label():
    return {"Del":"DAUC","Add":"IAUC","Spars":"Sparsity","IIC":"IIC","AD":"AD","ADD":"ADD","DelCorr":"DC","AddCorr":"IC","Time":"Time","Acc":"Accuracy"}
    
def latex_table_figure(exp_id,full=False,with_std=False,img_bckgr=False,suppMet=False):

    arr_dic = {}
    arr_f_dic = {}
    best_dic = {}

    what_is_best = get_what_is_best()

    if not suppMet:
        what_is_best.pop("Spars")
        what_is_best.pop("Time")

    metric_to_label = get_metric_label()
    metric_list = list(what_is_best.keys())

    for metric in metric_list:
        arr_dic[metric],arr_f_dic[metric],best_dic[metric] = loadArr(exp_id,metric,best=what_is_best[metric],img_bckgr=img_bckgr)

    id_to_label = get_id_to_label()

    res_list = []

    for i in range(len(arr_dic["Del"])):
       
        id = id_to_label[arr_dic["Del"][i,0]]

        mean = arr_f_dic["Del"][i].mean()
        dele,dele_std = round(mean,4),round(arr_f_dic["Del"][i].std(),3)
        dele_full_precision = mean
        is_best_dele = (mean==best_dic["Del"])

        res_dic = {"id":id,"Del_full_precision":dele_full_precision,
                    "Del":dele,  "Del_std":dele_std,  "is_best_Del":is_best_dele,}

        for metric in metric_list:
            if metric != "Del":
                mean_rnd,std_rnd,is_best = find_perf(i,metric,arr_dic,arr_f_dic,best_dic)
                res_dic.update({metric:mean_rnd,metric+"_std":std_rnd,"is_best_"+metric:is_best})

        res_list.append(res_dic)

    if full:
        res_list = addAccuracy(res_list,exp_id)
        metric_list.insert(8,"Acc")
        what_is_best["Acc"] = "max"

    res_list = sorted(res_list,key=lambda x:-x["Del_full_precision"])

    csv = "Model&Viz. Method&"+"&".join([metric_to_label[metric] for metric in metric_list])+"\\\\ \n"

    is_post_hoc = get_is_post_hoc()
    label_to_id = get_label_to_id()

    nbPostHoc = 0
    #Counting post hoc
    for i in range(len(res_list)):
        row = formatRow(res_list,i) 
        if is_post_hoc[label_to_id[row["id"]]]:
            nbPostHoc += 1 

    postHocInd =0
    #Adding post hoc
    for i in range(len(res_list)):
        row = formatRow(res_list,i) 
        if is_post_hoc[label_to_id[row["id"]]]:
            if postHocInd == 0:
                csv += "\multirow{"+str(postHocInd)+"}{*}{CNN}&"  
            else:
                csv += "&"
            csv += row["id"] 

            csv = processRow(csv,row,with_std,metric_list,full=full,postHoc=True,postHocInd=postHocInd,nbPostHoc=nbPostHoc)

            postHocInd += 1 

    csv += "\\hline \n"

    #Adding att models
    for i in range(len(res_list)):
        row = formatRow(res_list,i)
        if not is_post_hoc[label_to_id[row["id"]]]:
            csv += row["id"]+"&-"   
            csv = processRow(csv,row,with_std,metric_list)

    suff = "-IB" if img_bckgr else ""

    with open("../results/{}/attMetr_latex_table{}.csv".format(exp_id,suff),"w") as text:
        print(csv,file=text)

    for metric in metric_list:

        res_list = sorted(res_list,key=lambda x:x[metric])

        plt.figure()
        plt.errorbar(np.arange(len(res_list)),[row[metric] for row in res_list],[row[metric+"_std"] for row in res_list],color="darkblue",fmt='o')
        plt.bar(np.arange(len(res_list)),[row[metric] for row in res_list],0.9,color="lightblue",linewidth=0.5,edgecolor="darkblue")
        plt.ylabel(metric_to_label[metric])
        plt.ylim(bottom=0)
        plt.xticks(np.arange(len(res_list)),[row["id"] for row in res_list],rotation=45,ha="right")
        plt.tight_layout()
        plt.savefig("../vis/{}/attMetr_{}{}.png".format(exp_id,metric_to_label[metric],suff))

def latex_fmt(mean,std,is_best,with_std=False,with_start_char=True):

    if with_std:
        if is_best:
            metric_value = "\mathbf{"+str(mean)+"\pm"+str(std)+"}"
        else:
            metric_value = ""+str(mean)+"\pm"+str(std)
    else:
        if is_best:
            metric_value = "\mathbf{"+str(mean)+" }"
        else:
            metric_value = ""+str(mean)

    if with_start_char:
        metric_value = "&$" + metric_value + "$"
    
    return metric_value

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

def addAccuracy(res_list,exp_id):
    res_list_with_acc = []
    id_to_label= get_id_to_label()

    label_to_id = reverseLabDic(id_to_label,exp_id)

    acc_list = np.genfromtxt("../results/{}/attMetrics_Acc.csv".format(exp_id),delimiter=",",dtype=str)
    acc_dic = {model_id:{"mean":mean,"std":std} for (model_id,mean,std) in acc_list}

    for row in res_list:

        model_id = label_to_id[row["id"]]
        
        accuracy = float(acc_dic[model_id]["mean"])
        accuracy_std = float(acc_dic[model_id]["std"])

        row["Acc"] = str(round(accuracy,1))
        row["Acc_std"] = str(round(accuracy_std,1))
        row["Acc_full_precision"] = accuracy

        res_list_with_acc.append(row)
    
    bestAcc = max([row["Acc_full_precision"] for row in res_list_with_acc])

    res_list_with_acc_and_best = []
    for row in res_list_with_acc:   
        row["is_best_Acc"] = (row["Acc_full_precision"]==bestAcc)
        res_list_with_acc_and_best.append(row)

    return res_list_with_acc_and_best

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

def attCorrelation(exp_id,img_bckgr=False):

    if not os.path.exists("../vis/{}/correlation/".format(exp_id)):
        os.makedirs("../vis/{}/correlation/".format(exp_id))

    suff = "-IB" if img_bckgr else ""

    for metric in ["Del","Add"]:
        csv_res = []
        csv_res_pop = []

        paths = getResPaths(exp_id,metric,img_bckgr)

        for path in paths:
            points = np.load(path,allow_pickle=True).astype("float")
            
            path_att_score = path.replace("attMetr{}{}".format(metric,suff),"attMetrAttScore")
            if os.path.exists(path_att_score):
                model_id = os.path.basename(path).split(metric+suff+"_")[1].replace(".npy","")    

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

        suff = "-IB" if img_bckgr else ""

        with open(f"../results/{exp_id}/attMetrics_{metric}Corr{suff}.csv","w") as f:
            f.writelines(csv_res)

        with open(f"../results/{exp_id}/attMetrics_{metric}Corr{suff}_pop.csv","w") as f:
            f.writelines(csv_res_pop)

def attTime(exp_id):
    metric = "Time"

    csv_res = []
    csv_res_pop = []
    paths = sorted(glob.glob("../results/{}/attMetr{}_*.npy".format(exp_id,metric)))
    paths = list(filter(lambda x:os.path.basename(x).find("noise") ==-1,paths))
    paths = list(filter(lambda x:os.path.basename(x).find("imgBG") ==-1,paths))
    paths = list(filter(lambda x:os.path.basename(x).find("maskFeat") ==-1,paths))

    for path in paths:
        model_id = os.path.basename(path).split(metric+"_")[1].replace(".npy","")    
        all_times = np.load(path,allow_pickle=True)
        time = all_times.mean()
    
        csv_res += "{},{}\n".format(model_id,time) 
        csv_res_pop += "{},".format(model_id)+",".join([str(corr) for corr in all_times])+"\n"

    with open(f"../results/{exp_id}/attMetrics_{metric}.csv","w") as f:
        f.writelines(csv_res)

    with open(f"../results/{exp_id}/attMetrics_{metric}_pop.csv","w") as f:
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

def bar_viz(exp_id,img_bckgr=False):

    what_is_best = get_what_is_best()
    metric_list = list(what_is_best.keys())
    metric_list = list(filter(lambda metric:metric in ["Del","Add","AD","ADD","IIC"],metric_list))

    is_post_hoc = get_is_post_hoc()

    id_to_label = get_id_to_label()

    for metric in metric_list:

        perfs = loadPerf(exp_id,metric,img_bckgr=img_bckgr)

        perfs_att = []
        for perf in perfs:
            if not is_post_hoc[perf[0]]:
                perfs_att.append(perf)
        perfs_att = np.array(perfs_att)

        perfs_post = []
        for perf in perfs:
            if is_post_hoc[perf[0]]:
                perfs_post.append(perf)
        perfs_post = np.array(perfs_post)

        xmin,xmax = perfs[:,1].astype("float").min(),perfs[:,1].astype("float").max()

        fig = plt.figure(figsize=(5,2*len(perfs)))

        for i,model_perf in enumerate(perfs_att):
            subfig_mod = plt.subplot(len(perfs),1,i+1)
            fig.gca().axes.get_yaxis().set_visible(False)
            plt.xticks(fontsize=20)
            subfig_mod.hist(model_perf[1:].astype("float"),range=(xmin,xmax),color="orange",bins=10,label=id_to_label[model_perf[0]])
            plt.legend(prop={'size':20})

        for i,model_perf in enumerate(perfs_post):
            subfig_mod = plt.subplot(len(perfs),1,i+len(perfs_att)+1)
            fig.gca().axes.get_yaxis().set_visible(False)
            plt.xticks(fontsize=20)
            subfig_mod.hist(model_perf[1:].astype("float"),range=(xmin,xmax),color="blue",bins=10,label=id_to_label[model_perf[0]])
            plt.legend(prop={'size':20})

        plt.tight_layout()
        suff = "-IB" if img_bckgr else ""
        plt.savefig("../vis/{}/bar_{}{}.png".format(exp_id,metric,suff))

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
    cbar = plt.colorbar(cm.ScalarMappable(norm=matplotlib.colors.Normalize(-1,1),cmap=cmap))
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

    if not os.path.exists(path):
        if type(allFeat) is dict:
            for metric in allFeat:
                for bckgr in allFeat[metric]:
                    np.random.seed(0)
                    allFeat[metric][bckgr] = dimRedFunc(n_components=2,**kwargs).fit_transform(allFeat[metric][bckgr])
        else:
            allFeat = dimRedFunc(n_components=2,**kwargs).fit_transform(allFeat)

        np.save(path,allFeat)
    else:
        allFeat = np.load(path,allow_pickle=True)
        if len(allFeat.shape) == 0:
            allFeat = allFeat.item()

    return allFeat 

def dimred_metrics(exp_id,pop=False,dimred="umap",img_bckgr=False):

    metric_list = ["Del","DelCorr","ADD","Add","AddCorr","AD","IIC"] 
    if pop:
        metric_list.pop(-1)

    cmap = plt.get_cmap("Set1")
    colorInds = [0,4,5,1,2,3]
    colors = [cmap(colorInds[i]*1.0/8) for i in range(len(metric_list))]
    #colors = ["red","blue","green","violet","orange","yellow"]

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

def ood_repr(exp_id,quantitative,heavy_computation_models=False,metricList=None):

    backgrList = ["-IB","","-white","-gray","-blur"]
    if metricList is None:
        metricList = ["Lift","Del","Add"]

    os.makedirs(f"../vis/{exp_id}/representation_study/",exist_ok=True)

    file_paths = glob.glob(f"../results/{exp_id}/attMetrFeat*_*.npy")
    path_to_model_id = lambda x:"_".join(x.split("/")[-1].split("_")[1:]).replace(".npy","")
    model_ids = set(list(map(path_to_model_id,file_paths)))

    for model_id in model_ids:
        print(model_id)
        allFeat = {}
        for metric in metricList:
            allFeat[metric] = {}
            for backgr in backgrList:
                featPath = f"../results/{exp_id}/attMetrFeat{metric}{backgr}_{model_id}.npy"

                if os.path.exists(featPath):
                    feat = np.load(featPath,mmap_mode="r")

                    if metric in ["Add","Del"]:
                        sample_nb,step_nb = feat.shape[0],feat.shape[1]

                    allFeat[metric][backgr] = feat.reshape(feat.shape[0]*feat.shape[1],-1)
            
        umap_path = f"../results/{exp_id}/attMetrFeat_{model_id}.npy"
        if not quantitative:
            allFeat = run_dimred_or_load(umap_path,allFeat,dimred="umap")

        #We will train models to separate representations of altered images from representations of regular images
        if heavy_computation_models:
            constDic = {"NN":neural_network.MLPClassifier}
        else:
            constDic = {"SVM":svm.SVC,"DT":tree.DecisionTreeClassifier,"KNN":neighbors.KNeighborsClassifier}

        featNb = list(list(allFeat.values())[0].values())[0].shape[-1]

        kwargsDic = {"NN":{"hidden_layer_sizes":(128,),"batch_size":150 if metric=="Lift" else 2000,
                           "early_stopping":True,"learning_rate":"adaptive"},
                     "SVM":{"probability":True},"DT":{},"KNN":{}}

        metrDic = {"Del":(step_nb,plt.get_cmap("plasma"),lambda x:""),\
                   "Add":(step_nb,lambda x:plt.get_cmap("plasma")(1-x),lambda x:""),
                   "Lift":(3,lambda x:[plt.get_cmap("plasma")(0),"red","green"][int(x*step_nb_met)],lambda x:["Original Data","Mask","Inversed mask"][x])}

        for metric in metricList:
            print("\t",metric)
            step_nb_met,cmap,labels = metrDic[metric]

            for bckgr in backgrList:
                print("\t\t",bckgr)
                
                if bckgr in allFeat[metric]:
                    feat = allFeat[metric][bckgr].copy()
                    feat = feat.reshape(sample_nb,step_nb_met,-1)

                    if quantitative:
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
           
                        perfCSVPath = f"../results/{exp_id}/attMetrReprSep{metric}_{model_id}.csv"
                        if not os.path.exists(perfCSVPath):
                            with open(perfCSVPath,"w") as file:
                                print("bckgr,sec_model,train_acc,train_auc,val_acc,val_auc",file=file) 

                        for secModel in constDic.keys():
        
                            arr = np.genfromtxt(perfCSVPath,delimiter=",",dtype=str)
                            
                            if len(arr.shape) > 1:
                                inds = np.multiply(arr[:,0]==bckgr,arr[:,1]==secModel)
                                if len(inds) > 0:
                                    modelAlreadyTrained = len(np.argwhere(inds)[:,0]) > 0
                                else:
                                    modelAlreadyTrained = False 
                            #No model has been trained already 
                            #Or this model has not been already 
                            if len(arr.shape) == 1 or not modelAlreadyTrained: 
                                print(f"\t\t\tModel {secModel} being trained")                
                                model = constDic[secModel](**kwargsDic[secModel])
                                model.fit(train_x,train_y)
                                
                                train_y_score = model.predict_proba(train_x)[:,1]
                                train_acc = model.score(train_x,train_y)
                                train_auc = sklearn.metrics.roc_auc_score(train_y,train_y_score)
                                
                                val_y_score = model.predict_proba(val_x)[:,1]
                                val_acc = model.score(val_x,val_y)            
                                val_auc = sklearn.metrics.roc_auc_score(val_y,val_y_score)

                                with open(perfCSVPath,"a") as file:
                                    print(f"{bckgr},{secModel},{train_acc},{train_auc},{val_acc},{val_auc}",file=file)

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
                            plt.colorbar(cm.ScalarMappable(norm=matplotlib.colors.Normalize(0,1),cmap=plt.get_cmap("plasma")))
                        else:
                            plt.legend()

                        figPath = f"../vis/{exp_id}/representation_study/attMetrFeat{metric}{bckgr}_{model_id}.png"
                        plt.savefig(figPath)
                        plt.close()

def viz_ood_repr(exp_id):

    paths = glob.glob(f"../results/{exp_id}/attMetrReprSep*_*.csv")
    backgrList = ["-IB","","-white","-gray","-blur"]
    backgrLabels = {"-IB":"IB","":"Black","-white":"White","-gray":"Gray","-blur":"Blur"}

    sec_model_to_remove = ["GP","RF"]

    for path in paths:
        metric = path.split("/")[-1].split("ReprSep")[1].split("_")[0]
        model_id = path.split("/")[-1].split("ReprSep")[1].split("_")[1].replace(".csv","")

        csv = np.genfromtxt(path,delimiter=",",dtype=str)
        print(path)

        sec_model_list = list(set(csv[1:,1]))
        sec_model_list = list(filter(lambda x:x not in sec_model_to_remove,sec_model_list))
    
        for i in range(2):
            sec_model_metric_ind = csv.shape[1]-2+i
            sec_model_metric = csv[0,sec_model_metric_ind]

            perf_matrix = []
            for bckgr in backgrList:
                csv_bckgr = csv[1:][csv[1:,0] == bckgr]
                #print(csv_sec_model.shape,csv_rows.shape)
                 
                row = []
                for sec_model in sec_model_list:
                    csv_sec_model = csv_bckgr[(csv_bckgr[:,1] == sec_model)]
                    csv_sec_model_metric = csv_sec_model[:,sec_model_metric_ind]
                    if len(csv_sec_model_metric) == 0:
                        value = 0
                    else:
                        value = float(csv_sec_model_metric[0])

                    row.append(value)

                perf_matrix.append(row)
            
            perf_matrix = np.array(perf_matrix)
            print(perf_matrix)
            plt.figure()
            cmap_name = "plasma"
            cmap = plt.get_cmap(cmap_name)
            plt.imshow(perf_matrix,cmap=cmap_name)
            backLabelList = [backgrLabels[bckgr] for bckgr in backgrList]
            plt.yticks(np.arange(len(backgrList)),backLabelList)
            plt.xticks(np.arange(len(sec_model_list)),sec_model_list)
            for i in range(len(perf_matrix)):
                for j in range(len(perf_matrix[i])):
                    plt.text(j-.1,i,round(perf_matrix[i,j]*100,1))

            plt.colorbar(cm.ScalarMappable(norm=matplotlib.colors.Normalize(0,1),cmap=cmap))
            plt.tight_layout()
            plt.savefig(f"../vis/{exp_id}/representation_study/attMetrFeat{metric}_{model_id}_{sec_model_metric}.png")
    
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

def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--att_metrics',action="store_true") 
    argreader.parser.add_argument('--not_ignore_model',action="store_true") 
    argreader.parser.add_argument('--all_att_metrics',action="store_true") 
    argreader.parser.add_argument('--with_std',action="store_true") 
    argreader.parser.add_argument('--img_bckgr',action="store_true") 
    argreader.parser.add_argument('--pop',action="store_true")   
    argreader.parser.add_argument('--ood_repr',action="store_true") 
    argreader.parser.add_argument('--quantitative',action="store_true")  
    argreader.parser.add_argument('--heavy_computation_models',action="store_true")
    argreader.parser.add_argument('--viz_ood_repr',action="store_true")
    argreader.parser.add_argument('--dimred_metrics',action="store_true") 
    argreader.parser.add_argument('--dimred_func',type=str,default="tsne") 
    argreader.parser.add_argument('--ranking_similarities',action="store_true") 
    argreader.parser.add_argument('--find_best_methods',action="store_true") 
    argreader.parser.add_argument('--metrics',type=str,nargs="*") 

    ###################################### Accuracy per video ############################################""

    argreader.parser.add_argument('--accuracy_per_video',action="store_true") 
    
    argreader = load_data.addArgs(argreader)

    #Reading the comand line arg
    argreader.getRemainingArgs()

    #Getting the args from command line and config file
    args = argreader.args

    if args.att_metrics:

        suff = "-IB" if args.img_bckgr else ""

        for metric in ["Add","Del","Spars","Lift"]:
            attMetrics(args.exp_id,metric=metric,ignore_model=not args.not_ignore_model,img_bckgr=args.img_bckgr)
        
        #if not os.path.exists("../results/{}/attMetrics_AddCorr{}.csv".format(args.exp_id,suff)) or not os.path.exists("../results/{}/attMetrics_DelCorr{}.csv".format(args.exp_id,suff)):
        attCorrelation(args.exp_id,img_bckgr=args.img_bckgr)

        #if not os.path.exists("../results/{}/attMetrics_Time.csv".format(args.exp_id)):
        attTime(args.exp_id)
   
        for metric in ["Add","Del","Spars","IIC","AD","ADD","DelCorr","AddCorr","Time"]:
            ttest_attMetr(args.exp_id,metric=metric,img_bckgr=args.img_bckgr)

        bar_viz(args.exp_id,args.img_bckgr)

        latex_table_figure(args.exp_id,full=args.all_att_metrics,with_std=args.with_std,img_bckgr=args.img_bckgr)

    if args.ood_repr:
        ood_repr(args.exp_id,args.quantitative,args.heavy_computation_models,args.metrics)
    if args.viz_ood_repr:
        viz_ood_repr(args.exp_id)
    if args.dimred_metrics:
        dimred_metrics(args.exp_id,args.pop,args.dimred_func,args.img_bckgr)
    if args.ranking_similarities:
        ranking_similarities(args.exp_id,img_bckgr=args.img_bckgr,pop=args.pop)
    if args.find_best_methods:
        find_best_methods(args.exp_id,args.img_bckgr)

if __name__ == "__main__":
    main()
