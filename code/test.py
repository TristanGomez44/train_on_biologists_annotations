import glob,sys
import numpy as np

import subprocess

def csv_to_perf(path):
    return float(np.genfromtxt(path,dtype=str,delimiter=";")[-1].split(",")[0])

def main():

    result = subprocess.run(["python","trainVal.py","-c","model_cub10_test.config"])

    if result.returncode == 0:

        path_train = glob.glob(f"../results/CUB10/modelbr_npa_low_resolution_epoch1_metrics_train.csv")[0]
        path_val = glob.glob(f"../results/CUB10/modelbr_npa_low_resolution_epoch1_metrics_val.csv")[0]

        train_acc = csv_to_perf(path_train)
        val_acc = csv_to_perf(path_val)

        train_target = 0.16
        val_target = 0.40

        sucess = np.abs(train_acc - train_target) < 0.05
        sucess = sucess * (np.abs(val_acc - val_target) < 0.05)

        print(f"Train perf: {train_acc} (target={train_target}). Val perf: {val_acc} (target={val_target})")

    else:
        print(f"Script excecution failed: {result.stdout}")
        sucess=False

    if sucess:
        print(f"Test passed with sucess!")
    else:
        print(f"Test failed")

    #Must reach 16% training accuracy and 40% validation accuracy after one epoch

if __name__ == "__main__":
    main()