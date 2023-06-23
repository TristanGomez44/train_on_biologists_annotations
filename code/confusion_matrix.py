import glob

import numpy as np
import sklearn.metrics
import matplotlib.pyplot as plt 

from args import ArgReader,str2bool

def main(argv=None):
    # Getting arguments from config file and command line
    # Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--no_val', type=str2bool, help='To not compute the validation')
    argreader.parser.add_argument('--only_test', type=str2bool, help='To only compute the test')
    argreader.parser.add_argument('--val_freq', type=int, help='Frequency at which to run a validation.')

    argreader.parser.add_argument('--do_test_again', type=str2bool, help='Does the test evaluation even if it has already been done')

    argreader.parser.add_argument('--optuna', type=str2bool, help='To run a hyper-parameter study')
    argreader.parser.add_argument('--optuna_trial_nb', type=int, help='The number of hyper-parameter trial to run.')
    argreader.parser.add_argument('--opt_data_aug', type=str2bool, help='To optimise data augmentation hyper-parameter.')
    argreader.parser.add_argument('--opt_att_maps_nb', type=str2bool, help='To optimise the number of attention maps.')

    argreader.parser.add_argument('--max_batch_size', type=int, help='To maximum batch size to test.')

    argreader.parser.add_argument('--trial_id', type=int, help='The trial ID. Useful for grad exp during test')

    argreader.parser.add_argument('--log_gradient_norm_frequ', type=int, help='The step frequency at which to save gradient norm.')

    argreader.parser.add_argument('--save_output_during_validation', type=str2bool, help='To save model output during validation.')

    # Reading the comand line arg
    argreader.getRemainingArgs()

    args = argreader.args

    labels_csv = np.genfromtxt("../data/Blastocyst_Dataset/Gardner_test_gold_onlyGardnerScores.csv",delimiter=";",dtype=str)

    task_dic = {}
    for task in ["exp","icm","te"]:

        path = glob.glob(f"../results/{args.exp_id}/output_{task}_{args.model_id}_epoch*_test.npy")[0]
        output = np.load(path)

        pred = output.argmax(axis=-1)

        labels = labels_csv[1:,np.argwhere(labels_csv[0,:]==task.upper()+"_gold")[0][0]]

        labels[labels=="NA"] = "-1"
        labels[labels=="ND"] = "-1"

        labels_int = labels[labels!="-1"].astype("int")
        pred = pred[labels!="-1"].astype("int")    

        conf_mat = sklearn.metrics.confusion_matrix(labels_int,pred,normalize="pred")

        plt.figure()
        plt.imshow(conf_mat)

        plt.ylabel("True label")
        plt.xlabel("Predicted label")

        #Set ticks to integer values
        plt.xticks(np.arange(0,conf_mat.shape[1]),np.arange(0,conf_mat.shape[1]))
        plt.yticks(np.arange(0,conf_mat.shape[0]),np.arange(0,conf_mat.shape[0]))

        #Plot the value of the matrix
        for i in range(conf_mat.shape[0]):
            for j in range(conf_mat.shape[1]):
                plt.text(j, i, "{:.2f}".format(conf_mat[i, j]),
                        ha="center", va="center", color="w")

        plt.savefig(f"../vis/{args.exp_id}/conf_mat_{args.model_id}_{task}.png")
        plt.close()

        #Saves the metric values in a dictionnary 
        metrics = {}
        metrics["acc"] = sklearn.metrics.accuracy_score(labels_int,pred)
        occurency_of_most_frequent_class = np.bincount(labels_int).max()
        metrics["most_freq_acc"] = occurency_of_most_frequent_class/len(labels_int)
        metrics["recall"] = sklearn.metrics.recall_score(labels_int,pred,average="macro")
        metrics["precision"] = sklearn.metrics.precision_score(labels_int,pred,average="macro")
        metrics["f1"] = sklearn.metrics.f1_score(labels_int,pred,average="macro")
        
        #Saves the metrics in a csv file
        with open(f"../results/{args.exp_id}/metrics_{task}_{args.model_id}.csv","w") as f:
            f.write("metric,value\n")
            for key in metrics.keys():
                f.write(f"{key},{metrics[key]}\n")

        task_dic[task] = metrics

    metric_nb = task_dic[list(task_dic.keys())[0]].keys().__len__()

    #Saves task_dic in a .tex file 
    with open(f"../results/{args.exp_id}/metrics_{args.model_id}.tex","w") as f:
        f.write("\\begin{tabular}{c|c|"+"".join(["c" for _ in range(metric_nb-1)])+"}\n")
        f.write("\\hline\n")
        f.write("Task & Proportion of most frequent class & Accuracy & Recall & Precision & F1 \\\\\n")
        f.write("\\hline\n")

        for task in task_dic.keys():
            f.write(f"{task.upper()} & {task_dic[task]['most_freq_acc']:.2f} & {task_dic[task]['acc']:.2f} & {task_dic[task]['recall']:.2f} & {task_dic[task]['precision']:.2f} & {task_dic[task]['f1']:.2f} \\\\\n")
            f.write("\\hline\n")
        f.write("\\end{tabular}\n")

if __name__ == "__main__":
    main()