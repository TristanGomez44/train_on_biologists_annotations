import numpy as np
import matplotlib.pyplot as plt 
from scipy.stats import sem,t,bootstrap
import torch 

def mean_confidence_interval(a, confidence=0.95):
    n = a.shape[0]
    m, se = a.mean(axis=0), sem(a,axis=0)
    h = se * t.ppf((1 + confidence) / 2., n-1)
    return m, h


exp_id = "CROHN25"

background_func = lambda x:"blur" if x in ["Insertion"] else "black"

cmap = plt.get_cmap("rainbow")
nb_curves = 8

#plt.figure()

for cum_suff in ["","nc"]:

    fig, axs = plt.subplots(2,2)

    for i,model_id in enumerate(["noneRed2","noneRed_focal2"]):

        for j,metric in enumerate(["Deletion","Insertion"]):
            
            background = background_func(metric)
            metric += cum_suff

            #np.logspace(np.log10(0.01),np.log10(10),nb_curves)
            #np.linspace(0.01,10,nb_curves)
            for k,temp in enumerate([0.1,0.5,1,1.5,10]):

                dic = np.load(f"../results/{exp_id}/{metric}-{background}_{model_id}-gradcampp.npy",allow_pickle=True).item()
                scores = dic["prediction_scores"]

                scores = torch.softmax(torch.from_numpy(scores/temp),dim=-1)
                scores = scores.max(dim=-1)[0]
                #scores_mean = scores.mean(dim=0)
                #scores_std = scores.std(dim=0)

                #scores_mean,conf_interv = mean_confidence_interval(scores)
                print(scores.shape)
                conf_interv= np.zeros((2,scores.shape[1]))
                scores_mean = scores.mean(axis=0)
                for l in range(scores.shape[1]):
                    rng = np.random.default_rng(0)
                    
                    conf_interv[:,l] = np.quantile(scores[:,l],0.25),np.quantile(scores[:,l],0.75)

                    #if scores[:,l].min() != scores[:,l].max():
                    #    print(scores[:,l].min(),scores[:,l].max())
                    #    try:
                    #        res = bootstrap((scores[:,l],), np.mean, confidence_level=0.99,random_state=rng)
                    #        conf_interv[:,l] = res.confidence_interval.low,res.confidence_interval.high
                    #    except ValueError:
                    #        conf_interv[:,l] = scores_mean[l],scores_mean[l]
                    #else:
                    #    conf_interv[:,l] = scores_mean[l],scores_mean[l]

                x = np.arange(len(scores_mean))
                color = cmap(k*1.0/nb_curves)
                #axs[i,j].errorbar(x,scores_mean,yerr=conf_interv,color=cmap(k*1.0/nb_curves),alpha=0.25)
                axs[i,j].fill_between(x,conf_interv[0],conf_interv[1],color=color,alpha=0.25)
                axs[i,j].plot(x,scores_mean,label=round(temp,2),color=color)
                axs[i,j].set_ylim(0,1.1)
            
            axs[i,j].set_title(model_id+"_"+metric)

    axs[1,0].legend(bbox_to_anchor=(1.1, 1.05))
    plt.tight_layout()
    plt.savefig(f"../vis/{exp_id}/over_vs_under_confident{cum_suff}.png")
    plt.close()