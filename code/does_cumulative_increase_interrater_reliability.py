
from args import ArgReader
import os,sys,time
import glob
import numpy as np
import sqlite3 
from metrics import krippendorff_alpha,krippendorff_alpha_paralel,krippendorff_alpha_bootstrap
import matplotlib.pyplot as plt 
from scipy.stats._resampling import _bootstrap_iv,rng_integers,_percentile_of_score,ndtri,ndtr,_percentile_along_axis,BootstrapResult,ConfidenceInterval

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

def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    #Reading the comand line arg
    argreader.getRemainingArgs()

    #Getting the args from command line and config file
    args = argreader.args

    exp_id = "CROHN25"
    #model_ids = ["noneRed2","noneRed_onlyfocal","noneRed_onlylossonmasked","noneRed_focal2"]
    model_id = "noneRed_focal2"
    metrics = ["DAUC","DC","IAUC","IC"]
    background_func = lambda x:"blur" if x in ["IAUC","IC","INSERTION_VAL_RATE"] else "black"

    db_path = f"../results/{exp_id}/saliency_metrics.db"
   
    con = sqlite3.connect(db_path) # change to 'sqlite:///your_filename.db'
    cur = con.cursor()

    csv = "cumulative,"+ ",".join(metrics) + "\n"
    
    post_hoc_methods_list = []

    for cumulative_suff in ["","-nc"]:

        csv += "False" if cumulative_suff == "-nc" else "True"

        for metric in metrics:

            background = background_func(metric)
            metric += cumulative_suff
            output = cur.execute(f'SELECT post_hoc_method,metric_value FROM metrics WHERE model_id=="{model_id}" and metric_label=="{metric}" and replace_method=="{background}"').fetchall()

            post_hoc_methods,metric_values_list = zip(*output)

            metric_values_matrix = fmt_metric_values(metric_values_list)

            alpha = krippendorff_alpha_paralel(metric_values_matrix)
        
  
            post_hoc_methods_list.append(post_hoc_methods)

            #Bootstrap 
            rng = np.random.default_rng(0)
            method="bca"    
            res = bootstrap(metric_values_matrix, krippendorff_alpha_bootstrap, confidence_level=0.99,random_state=rng,method=method,vectorized=True,n_resamples=5000)
            confinterv= res.confidence_interval
            csv += ","+str(alpha)+" ("+str(confinterv.low)+" "+str(confinterv.high)+")"


        csv += "\n"

    with open(f"../results/{exp_id}/krippendorff_alpha.csv","w") as file:
        print(csv,file=file)

    post_hoc_methods_set = set(post_hoc_methods_list)

    if len(post_hoc_methods_set) != 1:
        print("Different sets of posthoc methods were used:",post_hoc_methods_set)

if __name__ == "__main__":
    main()
