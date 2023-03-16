
from args import ArgReader
import os,sys,time
import glob
import numpy as np
import sqlite3 
from metrics import krippendorff_alpha,krippendorff_alpha_paralel,krippendorff_alpha_bootstrap
import matplotlib.pyplot as plt 
from matplotlib.colors import Normalize
from matplotlib import cm
from scipy.stats._resampling import _bootstrap_iv,rng_integers,_percentile_of_score,ndtri,ndtr,_percentile_along_axis,BootstrapResult,ConfidenceInterval
from scipy.stats import pearsonr

def preprocc_matrix(metric_values_matrix,metric):

    #if metric in ["DC","IC"]:
    #    metric_values_matrix = (metric_values_matrix + 1)/2
    
    if metric == "IIC":
        metric_values_matrix = metric_values_matrix.astype("bool")

    return metric_values_matrix

def fmt_metric_values(metric_values_list):

    matrix = []

    for i in range(len(metric_values_list)):
        matrix.append(np.array(metric_values_list[i].split(";")).astype("float"))

    metric_values_matrix = np.stack(matrix,axis=0)

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
    theta_hat = np.asarray(statistic(*data, axis=axis))[..., None]
    percentile = _percentile_of_score(theta_hat_b, theta_hat, axis=-1)

    z0_hat = ndtri(percentile)

    # calculate a_hat
    theta_hat_ji = []  # j is for sample of data, i is for jackknife resample
    theta_hat_i = []
    for i in range(len(data)):
        inds =[j for j in range(len(data))]
        #print(inds,i)
        inds.remove(i)
        inds = np.array(inds).astype("int")
        data = np.array(data)
        data_jackknife = data[inds]
        theta_hat_i.append(statistic(*data_jackknife, axis=-1)[0])
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
        #resampled_data = []

        resampled_data = _bootstrap_resample(data, n_resamples=batch_actual,
                                        random_state=random_state)
        #resampled_data.append(resample)

        # Compute bootstrap distribution of statistic
        theta_hat_b.append(statistic(*resampled_data, axis=-1))
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
        theta_hat = statistic(*data, axis=-1)
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

def make_comparison_matrix(res_mat,p_val_mat,exp_id,labels,filename):

    p_val_mat = (p_val_mat<0.05)

    cmap = plt.get_cmap('bwr')

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
            if i < j:
                rad = 0.3 if p_val_mat[i,j] else 0.1
                circle = plt.Circle((i, j), rad, color=cmap(res_mat[i,j]))
                ax.add_patch(circle)

    fontsize = 17
    plt.yticks(np.arange(1,len(res_mat)),labels[1:],fontsize=fontsize)
    plt.xticks(np.arange(len(res_mat)-1),["" for _ in range(len(res_mat)-1)])
    
    for i in range(len(res_mat)-1):
        plt.text(i-0.2,i-0.4+1,labels[i],rotation=45,ha="left",fontsize=fontsize)
    plt.colorbar(cm.ScalarMappable(norm=Normalize(-1,1),cmap=cmap),orientation="vertical",pad=0.17,shrink=0.8,ax=ax)
    plt.tight_layout()
    plt.savefig(f"../vis/{exp_id}/{filename}")

def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--background', type=str)

    #Reading the comand line arg
    argreader.getRemainingArgs()

    #Getting the args from command line and config file
    args = argreader.args

    exp_id = "CROHN25"
    #model_ids = ["noneRed2","noneRed_onlyfocal","noneRed_onlylossonmasked","noneRed_focal2"]
    model_id = args.model_id
    #single_step_metrics = ["IIC","AD","ADD"]
    #multi_step_metrics = ["DAUC","DC","IAUC","IC"]

    #single_step_metrics = ["ADD"]
    #multi_step_metrics = ["IC","DC"]

    single_step_metrics = []
    multi_step_metrics = ["DC"]

    metrics = multi_step_metrics+single_step_metrics
    if args.background is None:
        background_func = lambda x:"blur" if x in ["IAUC","IC","INSERTION_VAL_RATE"] else "black"
    else:
        background_func = lambda x:args.background

    db_path = f"../results/{exp_id}/saliency_metrics.db"
   
    con = sqlite3.connect(db_path) # change to 'sqlite:///your_filename.db'
    cur = con.cursor()

    csv_krippen = "cumulative,"+ ",".join(metrics) + "\n"
    
    post_hoc_methods_list = []

    all_metric_values_matrix = []
    all_metric_values_matrix_cum = []

    for cumulative_suff in ["","-nc"]:

        csv_krippen += "False" if cumulative_suff == "-nc" else "True"

        for metric in metrics:
            
            print(metric,cumulative_suff)

            if metric not in single_step_metrics or cumulative_suff=="": 
                background = background_func(metric)
                metric += cumulative_suff
                output = cur.execute(f'SELECT post_hoc_method,metric_value FROM metrics WHERE model_id=="{model_id}" and metric_label=="{metric}" and replace_method=="{background}"').fetchall()

                post_hoc_methods,metric_values_list = zip(*output)

                metric_values_matrix = fmt_metric_values(metric_values_list)
                metric_values_matrix = preprocc_matrix(metric_values_matrix,metric)

                if cumulative_suff == "-nc":
                    all_metric_values_matrix.append(metric_values_matrix)
                else:
                    all_metric_values_matrix_cum.append(metric_values_matrix)

                #Krippendorf's alpha (inter-rater reliabiilty) 
                alpha = krippendorff_alpha_paralel(metric_values_matrix)
                rng = np.random.default_rng(0)
                res = bootstrap(metric_values_matrix, krippendorff_alpha_bootstrap, confidence_level=0.99,random_state=rng,method="bca" ,vectorized=True,n_resamples=5000)
                confinterv= res.confidence_interval
                csv_krippen += ","+str(alpha)+" ("+str(confinterv.low)+" "+str(confinterv.high)+")"

                plot_bootstrap_distr(res,exp_id,metric,cumulative_suff,"bca")

                '''
                #Inter-method reliability
                method_nb = metric_values_matrix.shape[1]
                inter_method_corr_mat = np.zeros((method_nb,method_nb))
                signif_mat = np.empty((method_nb,method_nb),dtype="bool")
                fig, ax = plt.subplots(method_nb,1,figsize=(10,14))
                for i in range(method_nb):
                    values_i = metric_values_matrix[:,i]
                    
                    ax[i].hist(values_i,label=post_hoc_methods[i])
                    ax[i].legend()
                    
                    for j in range(method_nb):
                        values_j = metric_values_matrix[:,j]
                        inter_method_corr_mat[i,j],signif_mat[i,j] = pearsonr(values_i,values_j)

                fig.tight_layout()
                fig.savefig(f"../vis/{exp_id}/{metric}.png")
                plt.close()

                make_comparison_matrix(inter_method_corr_mat,signif_mat,exp_id,post_hoc_methods,f"ttest_{metric}_{model_id}_b{background}.png")
                '''
                post_hoc_methods_list.append(post_hoc_methods)

        csv_krippen += "\n"

    with open(f"../results/{exp_id}/krippendorff_alpha_{model_id}_b{args.background}.csv","w") as file:
        print(csv_krippen,file=file)

    sys.exit(0)

    post_hoc_methods_set = set(post_hoc_methods_list)

    if len(post_hoc_methods_set) != 1:
        print("Different sets of posthoc methods were used:",post_hoc_methods_set)

    all_metric_values_matrix = np.stack(all_metric_values_matrix)
    all_metric_values_matrix_cum = np.stack(all_metric_values_matrix_cum)

    for i in range(all_metric_values_matrix.shape[2]):

        for cumulative in [True,False]:

            if cumulative:
                all_mat = all_metric_values_matrix_cum
            else:
                all_mat = all_metric_values_matrix
        
            metrics_ = metrics if cumulative else multi_step_metrics

            inter_metric_corr_mat = np.zeros((len(metrics_),len(metrics_)))
            signif_mat = np.empty((len(metrics_),len(metrics_)),dtype="bool")

            for j in range(len(metrics_)):

                for k in range(len(metrics_)):

                    inter_metric_corr_mat[j,k],signif_mat[j,k] = pearsonr(all_mat[j,:,i],all_mat[k,:,i])

            make_comparison_matrix(inter_metric_corr_mat,signif_mat,exp_id,metrics_,f"ttest_{post_hoc_methods[i]}_cum={cumulative}_{model_id}_b{background}.png")

if __name__ == "__main__":
    main()
