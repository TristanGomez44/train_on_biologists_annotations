from tkinter import Y
import sys
from args import ArgReader
import numpy as np
import sqlite3 
from metrics import krippendorff_alpha_paralel,krippendorff_alpha_bootstrap
import matplotlib.pyplot as plt 
from scipy.stats._resampling import _bootstrap_iv,rng_integers,_percentile_of_score,ndtri,ndtr,BootstrapResult,ConfidenceInterval

import warnings
from scipy.stats._warnings_errors import DegenerateDataWarning

from scipy.stats import pearsonr,kendalltau

def preprocc_matrix(metric_values_matrix,metric):

    if metric == "IIC":
        metric_values_matrix = metric_values_matrix.astype("bool")

    return metric_values_matrix

def fmt_value_str(value_str):

    value_list = value_str.split(";")

    if value_list[0].startswith("shape="):
        shape = value_list[0].replace("shape=(","").replace(")","").replace(",","").split(" ")
        shape = [int(size) for size in shape]

        value_list= value_list[1:]
    else:
        shape = (-1,)
    value_list = np.array(value_list).astype("float").reshape(shape)

    return value_list 

def fmt_metric_values(metric_values_list):
    matrix = []
    for i in range(len(metric_values_list)):
        matrix.append(fmt_value_str(metric_values_list[i]))
    metric_values_matrix = np.stack(matrix,axis=0)
    print(metric_values_matrix.shape)
    metric_values_matrix = metric_values_matrix.transpose(1,0)
    return metric_values_matrix

def _bootstrap_resample(sample, n_resamples=None, random_state=None):
    """Bootstrap resample the sample."""
    n = len(sample)

    # bootstrap - each row is a random resample of original observations
    i = rng_integers(random_state, 0, n, (n,))

    resamples = np.array(sample)[i]
    return resamples

def _bca_interval(data, statistic, axis, alpha, theta_hat_b, batch):
    """Bias-corrected and accelerated interval."""

    # closely follows [1] 14.3 and 15.4 (Eq. 15.36)

    # calculate z0_hat
    theta_hat = np.asarray(statistic(*data))[..., None]
    percentile = _percentile_of_score(theta_hat_b, theta_hat, axis=-1)

    z0_hat = ndtri(percentile)

    # calculate a_hat
    theta_hat_ji = []  # j is for sample of data, i is for jackknife resample
    theta_hat_i = []
    for i in range(len(data)):
        inds =[j for j in range(len(data))]
        inds.remove(i)
        inds = np.array(inds).astype("int")
        data = np.array(data)
        data_jackknife = data[inds]
        theta_hat_i.append(statistic(*data_jackknife)[0])
    theta_hat_ji.append(theta_hat_i)

    theta_hat_ji = [np.array(theta_hat_i)
                    for theta_hat_i in theta_hat_ji]

    n_j = [len(theta_hat_i) for theta_hat_i in theta_hat_ji]

    theta_hat_j_dot = [theta_hat_i.mean(axis=-1, keepdims=True)
                       for theta_hat_i in theta_hat_ji]

    U_ji = [(n - 1) * (theta_hat_dot - theta_hat_i)
            for theta_hat_dot, theta_hat_i, n
            in zip(theta_hat_j_dot, theta_hat_ji, n_j)]

    nums = [(U_i**3).sum(axis=-1)/n**3 for U_i, n in zip(U_ji, n_j)]
    dens = [(U_i**2).sum(axis=-1)/n**2 for U_i, n in zip(U_ji, n_j)]
    a_hat = 1/6 * sum(nums) / sum(dens)**(3/2)

    # calculate alpha_1, alpha_2
    z_alpha = ndtri(alpha)
    z_1alpha = -z_alpha
    num1 = z0_hat + z_alpha
    alpha_1 = ndtr(z0_hat + num1/(1 - a_hat*num1))
    num2 = z0_hat + z_1alpha
    alpha_2 = ndtr(z0_hat + num2/(1 - a_hat*num2))
    return alpha_1, alpha_2, a_hat  # return a_hat for testing

def _percentile_along_axis(theta_hat_b, alpha):
    """`np.percentile` with different percentile for each slice."""
    # the difference between _percentile_along_axis and np.percentile is that
    # np.percentile gets _all_ the qs for each axis slice, whereas
    # _percentile_along_axis gets the q corresponding with each axis slice
    shape = theta_hat_b.shape[:-1]

    alpha = np.broadcast_to(alpha[0], shape)
    percentiles = np.zeros_like(alpha, dtype=np.float64)
    for indices, alpha_i in np.ndenumerate(alpha):
        if np.isnan(alpha_i):
            # e.g. when bootstrap distribution has only one unique element
            msg = (
                "The BCa confidence interval cannot be calculated."
                " This problem is known to occur when the distribution"
                " is degenerate or the statistic is np.min."
            )
            warnings.warn(DegenerateDataWarning(msg))
            percentiles[indices] = np.nan
        else:
            theta_hat_b_i = theta_hat_b[indices]
            percentiles[indices] = np.percentile(theta_hat_b_i, alpha_i)
    return percentiles[()]  # return scalar instead of 0d array


def bootstrap(data, statistic, *, n_resamples=9999, batch=None,
              vectorized=None, paired=False, axis=0, confidence_level=0.95,
              method='BCa', bootstrap_result=None, random_state=None):
    # Input validation
    args = _bootstrap_iv(data, statistic, vectorized, paired, axis,
                         confidence_level, n_resamples, batch, method,
                         bootstrap_result, random_state)
    data, statistic, vectorized, paired, axis, confidence_level = args[:6]
    n_resamples, batch, method, bootstrap_result, random_state = args[6:]

    theta_hat_b = ([] if bootstrap_result is None
                   else [bootstrap_result.bootstrap_distribution])

    batch_nominal = batch or n_resamples or 1

    for k in range(0, n_resamples):
        batch_actual = min(batch_nominal, n_resamples-k)
        # Generate resamples

        resampled_data = _bootstrap_resample(data, n_resamples=batch_actual,
                                        random_state=random_state)

        # Compute bootstrap distribution of statistic
        theta_hat_b.append(statistic(*resampled_data))
    theta_hat_b = np.concatenate(theta_hat_b, axis=-1)

    # Calculate percentile interval
    alpha = (1 - confidence_level)/2
    if method == 'bca':
        interval = _bca_interval(data, statistic, axis=-1, alpha=alpha,
                                 theta_hat_b=theta_hat_b, batch=batch)[:2]
        percentile_fun = _percentile_along_axis
    else:
        interval = alpha, 1-alpha

        def percentile_fun(a, q):
            return np.percentile(a=a, q=q, axis=-1)

    # Calculate confidence interval of statistic
    ci_l = percentile_fun(theta_hat_b, interval[0]*100)
    ci_u = percentile_fun(theta_hat_b, interval[1]*100)
    if method == 'basic':  # see [3]
        theta_hat = statistic(*data)
        ci_l, ci_u = 2*theta_hat - ci_u, 2*theta_hat - ci_l

    return BootstrapResult(confidence_interval=ConfidenceInterval(ci_l, ci_u),
                           bootstrap_distribution=theta_hat_b,
                           standard_error=np.std(theta_hat_b, ddof=1, axis=-1))


def plot_bootstrap_distr(res,exp_id,metric,cumulative_suff,method):
    fig, ax = plt.subplots()
    ax.hist(res.bootstrap_distribution, bins=25)
    ax.set_title('Bootstrap Distribution')
    ax.set_xlabel('statistic value')
    ax.set_ylabel('frequency')
    plt.savefig(f"../vis/{exp_id}/bootstrap_distr_{metric}{cumulative_suff}_{method}.png")
    plt.close()

def get_post_hoc_label_dic():
    return {"ablationcam":"Ablation-CAM","gradcam":"Grad-CAM","gradcampp":"Grad-CAM++","scorecam":"Score-CAM"}

def get_model_label_dic():
    return {"noneRed2":"ResNet50","noneRed_focal2":"ResNet50-FL","noneRed2_transf":"ViT-b16","noneRed_focal2_transf":"ViT-b16-FL"}

def make_comparison_matrix(res_mat,p_val_mat,exp_id,labels,filename,axs,subplot_row,subplot_col,label=None,fontsize=17):

    label_dic = get_post_hoc_label_dic()
    label_dic.update(get_model_label_dic())

    p_val_mat = (p_val_mat<0.05)

    cmap = plt.get_cmap('bwr')
    
    ax = axs[subplot_row,subplot_col]
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])

    ax.imshow(p_val_mat*0,cmap="Greys")
    for i in range(len(res_mat)):
        for j in range(len(res_mat)):
            if i < j:
                rad = 0.4 if p_val_mat[i,j] else 0.1

                circle = plt.Circle((i, j), rad+0.05, color="black")
                ax.add_patch(circle)
                if p_val_mat[i,j]:
                    circle = plt.Circle((i, j), rad, color=cmap(res_mat[i,j]*0.5+0.5))
                    ax.add_patch(circle)

                    color = "white" if abs(res_mat[i,j]) > 0.6 else "black"

                    ax.text(i,j+0.1,round(res_mat[i,j]*100),ha="center",fontsize=fontsize,color=color)                

   
    labels = [label_dic[label] if label in label_dic else label for label in labels]

    ax.set_yticks(np.arange(1,len(res_mat)),labels[:-1],fontsize=fontsize)
    ax.set_xticks(np.arange(len(res_mat)-1),["" for _ in range(len(res_mat)-1)])
    for i in range(len(res_mat)-1):
        ax.text(i-0.2,i-0.5+1,labels[i+1],rotation=45,ha="left",fontsize=fontsize)

    if not label is None:
        if label in label_dic:
            label = label_dic[label]
        ax.set_title(label,y=-0.2,fontsize=20)

    plt.tight_layout()
    plt.savefig(f"../vis/{exp_id}/{filename}")

def scattter(j,k,j_inds,k_inds,values_j,values_k,labels,file_id):
    if j in j_inds and k in k_inds and (j!=k):
        plt.figure(j*100+k)
        plt.scatter(values_j,values_k)
        plt.ylabel(labels[k])
        plt.xlabel(labels[j])
        plt.savefig(f"../vis/CROHN25/scatter_{labels[k]}_{labels[j]}_{file_id}.png")
        plt.close()

def get_metric_ind(metric):
    order_dic = {"DAUC":1,"DC":2,"ADD":3,"IAUC":4,"IC":5,"AD":6}

    order = 0

    if "-nc" in metric:
        order += 0.5
        metric = metric.split("-")[0]

    order = order_dic[metric]


    return order
    
def sort_lerf_from_morf(metrics,all_mat):
    key= lambda x:get_metric_ind(x[0])
    metrics_and_all_mat = zip(metrics,all_mat)
    metrics_and_all_mat = sorted(metrics_and_all_mat,key=key)
    metrics,all_mat = zip(*metrics_and_all_mat)
    
    all_mat = np.stack(all_mat)

    return metrics,all_mat

def make_krippen_bar_plot(array_krippen,array_krippen_err,metrics,multi_step_metrics,exp_id,filename_suff):

    width = 0.2
    fontsize = 17 

    plt.figure(-1)

    x = np.arange(len(metrics))
    y = array_krippen[0]
    yerr = array_krippen_err[0].transpose(1,0)
    yerr = np.abs(y[np.newaxis] - yerr)

    plt.bar(x,y,width=width,label="Cumulative")
    plt.errorbar(x,y,yerr,fmt='none',color="black")

    plt.ylabel("Krippendorf's alpha",fontsize=fontsize)
    plt.xticks(x,metrics,rotation=45,fontsize=fontsize,ha="right")
    yticks = np.arange(10)/10
    plt.yticks(yticks,yticks,fontsize=fontsize)
    plt.xlim(-0.5,len(metrics)+0.5)

    for threshold,label in zip([0.667,0.8],["Minimum reliability","Acceptable reliability"]):
        plt.plot([-0.5,len(metrics)],[threshold,threshold],"--",color="black")
        plt.text(-0.25,threshold+0.01,label)

    x = np.arange(len(multi_step_metrics))
    y = array_krippen[1,:len(x)]
    yerr = array_krippen_err[1,:len(x)].transpose(1,0)
    yerr = np.abs(y[np.newaxis] - yerr)
    plt.bar(x+width,y,width=width,label="Non cumulative")
    plt.errorbar(x+width,y,yerr,fmt='none',color="black")

    plt.legend()
    plt.tight_layout()

    plt.savefig(f"../vis/{exp_id}/krippendorf_{filename_suff}.png")
    plt.close()

def inter_method_reliability(metric_values_matrix,corr_func,exp_id,labels,filename_suff,axs,subplot_ind,cumulative_suff,subplot_name):
    method_nb = metric_values_matrix.shape[1]
    inter_method_corr_mat = np.zeros((method_nb,method_nb))
    signif_mat = np.empty((method_nb,method_nb))
    for i in range(method_nb):
        values_i = metric_values_matrix[:,i]
            
        for j in range(method_nb):
            values_j = metric_values_matrix[:,j]
            inter_method_corr_mat[i,j],signif_mat[i,j] = corr_func(values_i,values_j)

    make_comparison_matrix(inter_method_corr_mat,signif_mat,exp_id,labels,f"ttest_intermethod_{filename_suff}.png",axs,1*(cumulative_suff=="-nc"),subplot_ind,subplot_name)

def krippendorf(metric_values_matrix_alpha,exp_id,metric,cumulative_suff,csv_krippen,array_krippen,array_krippen_err,metr_ind):

    alpha = krippendorff_alpha_paralel(metric_values_matrix_alpha)
    rng = np.random.default_rng(0)
    res = bootstrap(metric_values_matrix_alpha, krippendorff_alpha_bootstrap, confidence_level=0.99,random_state=rng,method="bca" ,vectorized=True,n_resamples=50)
    confinterv= res.confidence_interval
    csv_krippen += ","+str(alpha)+" ("+str(confinterv.low)+" "+str(confinterv.high)+")"

    array_krippen[1*(cumulative_suff == "-nc"),metr_ind] = alpha
    array_krippen_err[1*(cumulative_suff == "-nc"),metr_ind] = np.array([confinterv.low,confinterv.high])

    plot_bootstrap_distr(res,exp_id,metric,cumulative_suff,"bca")

    return csv_krippen, array_krippen, array_krippen_err

def inner_reliability(metrics,multi_step_metrics,all_metric_values_matrix,all_metric_values_matrix_cum,corr_func,exp_id,filename_suff,explanation_names):

    all_metric_values_matrix = np.stack(all_metric_values_matrix)
    all_metric_values_matrix_cum = np.stack(all_metric_values_matrix_cum)
    
    plot_nb= all_metric_values_matrix.shape[2]

    fig, axs = plt.subplots(2,plot_nb//2,figsize=(20,10))
    
    all_mat = np.concatenate((all_metric_values_matrix_cum,all_metric_values_matrix),axis=0)

    metrics.remove("IIC")
    metrics_ = metrics + [metric+"-nc" for metric in multi_step_metrics]

    metrics_,all_mat = sort_lerf_from_morf(metrics_,all_mat)

    for i in range(all_metric_values_matrix.shape[2]):

        inter_metric_corr_mat = np.zeros((len(metrics_),len(metrics_)))
        signif_mat = np.empty((len(metrics_),len(metrics_)))

        for j in range(all_mat.shape[0]):
            for k in range(all_mat.shape[0]):
                inter_metric_corr_mat[j,k],signif_mat[j,k] = corr_func(all_mat[j,:,i],all_mat[k,:,i])

        make_comparison_matrix(inter_metric_corr_mat,signif_mat,exp_id,metrics_,f"ttest_intermetric_{filename_suff}.png",axs,i//2,i%2,explanation_names[i],fontsize=12)

def get_background_func(background):
    if background is None:
        background_func = lambda x:"blur" if x in ["IAUC","IC","INSERTION_VAL_RATE"] else "black"
    else:
        background_func = lambda x:background
    return background_func 

def addArgs(argreader):
    argreader.parser.add_argument('--background', type=str)
    argreader.parser.add_argument('--ordinal_metric', action="store_true")
    argreader.parser.add_argument('--compare_models', action="store_true")
    argreader.parser.add_argument('--accepted_models_to_compare',nargs="*",type=str)
    return argreader

def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    argreader = addArgs(argreader)

    #Reading the comand line arg
    argreader.getRemainingArgs()

    #Getting the args from command line and config file
    args = argreader.args

    exp_id = "CROHN25"

    model_id = args.model_id

    if args.compare_models:
        if args.accepted_models_to_compare is None:
            accepted_models = ["noneRed2","noneRed2_transf","noneRed_focal2","noneRed_focal2_transf"]
        else:
            accepted_models = args.accepted_models_to_compare
    else:
        accepted_models = None

    single_step_metrics = ["IIC","AD","ADD"]
    multi_step_metrics = ["DAUC","DC","IAUC","IC"]
    metrics_to_minimize = ["DAUC","AD"]
    metrics = multi_step_metrics+single_step_metrics

    background_func = get_background_func(args.background)

    db_path = f"../results/{exp_id}/saliency_metrics.db"
   
    con = sqlite3.connect(db_path) # change to 'sqlite:///your_filename.db'
    cur = con.cursor()

    csv_krippen = "cumulative,"+ ",".join(metrics) + "\n"
    array_krippen=np.zeros((2,len(metrics)))
    array_krippen_err=np.zeros((2,len(metrics),2))

    explanation_names_list = []

    all_metric_values_matrix = []
    all_metric_values_matrix_cum = []

    if args.ordinal_metric:
        corr_func = kendalltau
    else:
        corr_func = pearsonr

    if args.compare_models:
        filename_suff = f"models"
    else:
        filename_suff = f"{model_id}_b{args.background}"

    fig, axs = plt.subplots(2,len(metrics),figsize=(35,15))

    for cumulative_suff in ["","-nc"]:

        csv_krippen += "False" if cumulative_suff == "-nc" else "True"

        for metr_ind,metric in enumerate(metrics):
            
            print(metric,cumulative_suff)

            if metric not in single_step_metrics or cumulative_suff=="": 
                background = background_func(metric)
                metric += cumulative_suff

                if args.compare_models:
                    query = f'SELECT model_id,metric_value FROM metrics WHERE post_hoc_method=="" and metric_label=="{metric}" and replace_method=="{background}"'    
                else:
                    query = f'SELECT post_hoc_method,metric_value FROM metrics WHERE model_id=="{model_id}" and metric_label=="{metric}" and replace_method=="{background}"'
                
                output = cur.execute(query).fetchall()

                if args.compare_models:
                    output = list(filter(lambda x:x[0] in accepted_models,output))
                else:
                    output = list(filter(lambda x:x[0] != "",output))
                
                explanation_names,metric_values_list = zip(*output)

                metric_values_matrix = fmt_metric_values(metric_values_list)
                metric_values_matrix = preprocc_matrix(metric_values_matrix,metric)

                if args.ordinal_metric:
                    if metric not in metrics_to_minimize and metric != "IIC":
                        metric_values_matrix = -metric_values_matrix
                    
                    metric_values_matrix_alpha = metric_values_matrix.argsort(-1)+1
                else:
                    metric_values_matrix_alpha = metric_values_matrix

                #Krippendorf's alpha (inter-rater reliabiilty) 
                csv_krippen, array_krippen, array_krippen_err = krippendorf(metric_values_matrix_alpha,exp_id,metric,cumulative_suff,csv_krippen,array_krippen,array_krippen_err,metr_ind)

                if metric != "IIC":
                    #Inter-method reliability
                    inter_method_reliability(metric_values_matrix,corr_func,exp_id,explanation_names,filename_suff,axs,metr_ind,cumulative_suff,metric)

                    if cumulative_suff == "-nc":
                        all_metric_values_matrix.append(metric_values_matrix)
                    else:
                        all_metric_values_matrix_cum.append(metric_values_matrix)

                explanation_names_list.append(explanation_names)

        csv_krippen += "\n"

    with open(f"../results/{exp_id}/krippendorff_alpha_{filename_suff}.csv","w") as file:
        print(csv_krippen,file=file)

    make_krippen_bar_plot(array_krippen,array_krippen_err,metrics,multi_step_metrics,exp_id,filename_suff)

    explanation_names_set = set(explanation_names_list)

    if len(explanation_names_set) != 1:
        print("Different sets of explanations methods were used:",explanation_names_set)

    inner_reliability(metrics,multi_step_metrics,all_metric_values_matrix,all_metric_values_matrix_cum,corr_func,exp_id,filename_suff,explanation_names)

if __name__ == "__main__":
    main()