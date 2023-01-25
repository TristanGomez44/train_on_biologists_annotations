import os,shutil,sys

import numpy as np

from args import ArgReader

def main(argv=None):
    # Getting arguments from config file and command line
    # Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--dataset_path', type=str)

    # Reading the comand line arg
    argreader.getRemainingArgs()

    args = argreader.args

    root = args.dataset_path

    csv_path = os.path.join(root,"CrohnIPI_description.csv")

    arr = np.genfromtxt(csv_path,dtype=str,delimiter=",")[1:]

    #Removing the '>' character
    arr[:,1] = np.array(list(map(lambda x:x.replace(">",""),arr[:,1])))

    fold_nb = 5 #The number of split is supposed to be 5
    classes = list(set(arr[:,1]))

    class_dic = {row[0]:{"class":row[1],"split":int(row[2])} for row in arr}

    for fold_ind in range(1,fold_nb+1):
        for data_set in ["train","val","test"]:
            os.makedirs(f"../data/crohn_{fold_ind}_{data_set}/",exist_ok=True)

            for class_name in classes:
                os.makedirs(f"../data/crohn_{fold_ind}_{data_set}/{class_name}/",exist_ok=True)

    splits = np.arange(fold_nb)

    for i,frame_name in enumerate(class_dic):
        class_name,split = class_dic[frame_name]["class"],class_dic[frame_name]["split"]

        source = os.path.join(root,"imgs",frame_name)

        for fold_ind in range(1,fold_nb+1):

            splits_shifted = ((splits + (fold_ind - 1)) % fold_nb) + 1

            if split in list(splits_shifted[:3]):
                data_set = "train"
            elif split == splits_shifted[3]:
                data_set = "val"
            else:
                data_set = "test"

            destination = f"../data/crohn_{fold_ind}_{data_set}/{class_name}/{frame_name}"

            shutil.copy(source,destination)

if __name__ == "__main__":
    main()