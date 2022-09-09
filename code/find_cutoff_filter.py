
import args
from args import ArgReader
import os 

import torch,numpy as np

import trainVal,load_data

def main(argv=None):
    # Getting arguments from config file and command line
    # Building the arg reader
    argreader = ArgReader(argv)
    argreader.parser.add_argument('--att_metrics_img_nb', type=int, help='The nb of images on which to compute the att metric.')

    argreader = load_data.addArgs(argreader)

    # Reading the comand line arg
    argreader.getRemainingArgs()
    # Update the config args
    args = argreader.args

    args.val_batch_size = 1
    args.dataset_test = "embryo_img_train"
    _,testDataset = load_data.buildTestLoader(args, "test",withSeg=False)

    torch.manual_seed(0)
    #inds = torch.randint(len(testDataset),size=(1000,))

    avg_fft_path = "../results/EMB10/avg_fft.npy"

    if not os.path.exists(avg_fft_path):
        avg_fft = np.zeros_like(trainVal.getBatch(testDataset,0,args)[0].cpu(),dtype="complex64")[0,0]

        print("Computing the average FFT")
        for i in range(len(testDataset)):
            if i % 2000 == 0:
                print(i,"/",len(testDataset))
        
            data = trainVal.getBatch(testDataset,i,args)[0]

            fft_data = torch.fft.fft2(data[0,0])
            fft_shift_data = np.fft.fftshift(fft_data.cpu())

            avg_fft += fft_shift_data * 1.0 / len(testDataset)
        
        np.save(avg_fft_path,avg_fft)
    else:
        print("Loading avg FFT")
        avg_fft = np.load(avg_fft_path)

    avg_fft = avg_fft[0,0]

    print("Finding cutoff frequency")

    start=0
    end=1 
    cutoff_freq_found=False 
    prec_partial_sum = None

    while not cutoff_freq_found:
        partial_sum = compute_partial_sum((start+end)*0.5,avg_fft)

        print(start,end,partial_sum,np.abs(avg_fft).sum()*0.5)

        if partial_sum < np.abs(avg_fft).sum()*0.5:
            start = (start+end)*0.5
        else:
            end = (start+end)*0.5
        
        if prec_partial_sum == partial_sum:
            cutoff_freq_found = True 
        else:
            prec_partial_sum = partial_sum
    #print(start,partial_sum,np.abs(avg_fft).sum()*0.5)

    with open("../results/EMB10/cutoff_filter_freq.csv","w") as file:
        print(cutoff_freq_found,file=file)

def compute_partial_sum(cutoff_freq,fft):
    cx,cy = fft.shape[1]//2,fft.shape[0]//2
    rx,ry = int(cutoff_freq*cx*0.5),int(cutoff_freq*cy*0.5)

    mask = np.ones_like(fft,dtype="float64")
    mask[cy-ry:cy+ry,cx-rx:cx+rx] = 0
    mask = 1 - mask

    return np.abs(fft[mask.astype("bool")]).sum()
    
if __name__ =="__main__":
    main()