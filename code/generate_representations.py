
import glob 

import numpy as np 
import torch,torchvision

from trainVal import addInitArgs,addValArgs,getBatch,preprocessAndLoadParams
import modelBuilder,load_data
import args
from args import ArgReader
from args import str2bool


def apply_transf(data,transf):
    if transf == "identity":
        return data
    elif transf == "black_patches":

        #min_patch_nb = 10
        #max_patch_nb = 100 
        #nb_patches = np.random.randint(min_patch_nb,max_patch_nb,size=(1,))[0]
        nb_patches = 10
        min_size = 10 
        max_size = 100

        sizes = np.random.randint(min_size,max_size,size=(nb_patches))
        x = np.random.randint(0,data.shape[2]-1,size=(nb_patches))
        y = np.random.randint(0,data.shape[3]-1,size=(nb_patches))

        for i in range(nb_patches):
            data[:,:,x[i]:x[i]+sizes[i],y[i]:y[i]+sizes[i]] = 0

        return data 

    else:
        raise ValueError("Unkown transformation function",transf)

def main(argv=None):
    # Getting arguments from config file and command line
    # Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--transf', type=str,default="identity")
    argreader.parser.add_argument('--att_metrics_img_nb', type=int, help='The nb of images on which to compute the att metric.')
   
    argreader = addInitArgs(argreader)
    argreader = addValArgs(argreader)

    argreader = modelBuilder.addArgs(argreader)
    argreader = load_data.addArgs(argreader)

    # Reading the comand line arg
    argreader.getRemainingArgs()

    args = argreader.args

    args.cuda = args.cuda and torch.cuda.is_available()

    bestPath = glob.glob("../models/{}/model{}_best_epoch*".format(args.exp_id, args.model_id))[0]
    net = modelBuilder.netBuilder(args,gpu=0)
    net = preprocessAndLoadParams(bestPath,args.cuda,net,args.strict_init)
    net.eval()
    
    args.val_batch_size = 1
    testLoader,testDataset = load_data.buildTestLoader(args, "test",withSeg=args.with_seg)

    torch.manual_seed(0)
    inds = torch.randint(len(testDataset),size=(args.att_metrics_img_nb,))

    featList = []
    with torch.no_grad():
        for imgInd,i in enumerate(inds):
            if imgInd % 20 == 0 :
                print("Img",i.item(),"(",imgInd,"/",len(inds),")")

            data,targ = getBatch(testDataset,i,args)

            data_transf = apply_transf(data,args.transf)

            resDic = net(data_transf)

            featList.append(resDic["x"])
        
    featList = torch.cat(featList,axis=0).cpu().numpy()
    np.save(f"../results/{args.exp_id}/img_repr_model{args.model_id}_transf{args.transf}.npy",featList)

if __name__ == "__main__":
    main()