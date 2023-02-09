import os
import glob
from tkinter import E

import numpy as np
import torch

from args import ArgReader
import modelBuilder
import load_data
from compute_scores_for_saliency_metrics import getAttMetrMod,applyPostHoc
from trainVal import addInitArgs,preprocessAndLoadParams,init_post_hoc_arg

def main(argv=None):
    # Getting arguments from config file and command line
    # Building the arg reader
    argreader = ArgReader(argv)

    argreader = init_post_hoc_arg(argreader)
    argreader.parser.add_argument('--viz_id', type=str, help='The visualization id.',default="")
    argreader.parser.add_argument('--inds', type=int, help='The indexes of the images to use for computing att maps.',nargs="*")
    argreader = modelBuilder.addArgs(argreader)
    argreader = load_data.addArgs(argreader)
    argreader = addInitArgs(argreader)
    # Reading the comand line arg
    argreader.getRemainingArgs()

    args = argreader.args

    args.val_batch_size = 1
    _,testDataset = load_data.buildTestLoader(args, "test")

    bestPath = glob.glob("../models/{}/model{}_best_epoch*".format(args.exp_id, args.model_id))[0]
    bestEpoch = int(os.path.basename(bestPath).split("epoch")[1])

    net = modelBuilder.netBuilder(args)
    net = preprocessAndLoadParams(bestPath,args.cuda,net)
    net.eval()
        
    attrFunc,kwargs = getAttMetrMod(net,testDataset,args)

    if len(args.inds) == 0:
        args.inds = np.arange(len(testDataset))
    saliency_maps = []

    torch.set_grad_enabled(args.att_metrics_post_hoc == "gradcam_pp")

    for i in args.inds:
        batch = testDataset.__getitem__(i)
        data,targ = batch[0].unsqueeze(0),torch.tensor(batch[1]).unsqueeze(0)

        if args.cuda and torch.cuda.is_available():
            data = data.cuda()
            targ = targ.cuda()

        saliency_map = applyPostHoc(attrFunc,data,targ,kwargs,args)
        saliency_maps.append(saliency_map)

    saliency_maps = torch.cat(saliency_maps,dim=0).cpu()
    np.save(f"../results/{args.exp_id}/saliencymaps_{args.att_metrics_post_hoc}_{args.model_id}_epoch{bestEpoch}_{args.viz_id}.npy",saliency_maps.numpy())

if __name__ == "__main__":
    main()