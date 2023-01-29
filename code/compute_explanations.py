


import os
import sys
import glob

import numpy as np
import torch
from torch.nn import functional as F
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import captum
from captum.attr import (IntegratedGradients,NoiseTunnel)

import args
from args import ArgReader
from args import str2bool
import modelBuilder
import load_data
import metrics
from gradcam import GradCAMpp
from score_map import ScoreCam
from rise import RISE
from trainVal import addInitArgs,preprocessAndLoadParams

def main(argv=None):
    # Getting arguments from config file and command line
    # Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--grad_cam', type=int, help='To compute grad cam instead of training or testing.',nargs="*")
    argreader.parser.add_argument('--rise', type=str2bool, help='To compute rise instead or gradcam')
    argreader.parser.add_argument('--rise_resolution', type=int, help='',default=7)
    argreader.parser.add_argument('--score_map', type=str2bool, help='To compute score_map instead or gradcam')
    argreader.parser.add_argument('--noise_tunnel', type=str2bool, help='To compute the methods based on noise tunnel instead or gradcam')

    argreader.parser.add_argument('--viz_id', type=str, help='The visualization id.',default="")

    argreader = addInitArgs(argreader)
    argreader = modelBuilder.addArgs(argreader)
    argreader = load_data.addArgs(argreader)

    # Reading the comand line arg
    argreader.getRemainingArgs()

    args = argreader.args

    args.val_batch_size = 1
    testLoader,testDataset = load_data.buildTestLoader(args, "test")

    bestPath = glob.glob("../models/{}/model{}_best_epoch*".format(args.exp_id, args.model_id))[0]
    bestEpoch = int(os.path.basename(bestPath).split("epoch")[1])

    net = modelBuilder.netBuilder(args)
    net_raw = preprocessAndLoadParams(bestPath,args.cuda,net)

    if not args.rise and not args.score_map and not args.noise_tunnel:
        net = modelBuilder.GradCamMod(net_raw)
        model_dict = dict(type=args.first_mod, arch=net, layer_name='layer4', input_size=(448, 448))
        grad_cam = captum.attr.LayerGradCam(net.forward,net.layer4)
        grad_cam_pp = GradCAMpp(model_dict,True)
        guided_backprop_mod = captum.attr.GuidedBackprop(net)
        
        allMask = None
        allMask_pp = None
        allMaps = None 
    elif args.score_map:
        score_mod = ScoreCam(net_raw)
        allScore = None
    elif args.noise_tunnel:
        torch.backends.cudnn.benchmark = False

        net_raw.eval()
        net = modelBuilder.GradCamMod(net_raw)

        ig = IntegratedGradients(net)
        nt = NoiseTunnel(ig)

        batch = testDataset.__getitem__(0)
        data_base = torch.zeros_like(batch[0].unsqueeze(0))
        if args.cuda:
            data_base = data_base.cuda()
        
        allSq = None 
        allVar = None
    else:
        rise_mod = RISE(net_raw,res=args.rise_resolution)
        allRise = None

    if args.grad_cam == [-1]:
        args.grad_cam = np.arange(len(testDataset))
    
    for i in args.grad_cam:
        batch = testDataset.__getitem__(i)
        data,targ = batch[0].unsqueeze(0),torch.tensor(batch[1]).unsqueeze(0)

        if args.cuda:
            data = data.cuda()
            targ = targ.cuda()

        if not args.rise and not args.score_map and not args.noise_tunnel:

            mask = grad_cam.attribute(data,targ).detach().cpu()
            mask_pp = grad_cam_pp(data,targ).detach().cpu()
            map = guided_backprop_mod.attribute(data,targ).detach().cpu()

            if allMask is None:
                allMask = mask
                allMask_pp = mask_pp
                allMaps = map
            else:
                allMask = torch.cat((allMask,mask),dim=0)
                allMask_pp = torch.cat((allMask_pp,mask_pp),dim=0)
                allMaps = torch.cat((allMaps,map),dim=0)
        elif args.score_map:
            score_map = score_mod.generate_cam(data).detach().cpu()

            if allScore is None:
                allScore = torch.tensor(score_map)
            else:
                allScore = torch.cat((allScore,torch.tensor(score_map)),dim=0)       
        elif args.noise_tunnel:    

            attr_sq = nt.attribute(data, nt_type='smoothgrad_sq', stdevs=0.02, nt_samples=16,nt_samples_batch_size=3,baselines=data_base, target=targ)
            attr_var = nt.attribute(data, nt_type='vargrad', stdevs=0.02, nt_samples=16,nt_samples_batch_size=3,baselines=data_base, target=targ)

            attr_sq,attr_var = attr_sq.detach().cpu(),attr_var.detach().cpu()

            if allSq is None:
                allSq,allVar = attr_sq,attr_var
            else:
                allSq,allVar = torch.cat((allSq,attr_sq),dim=0),torch.cat((allVar,attr_var),dim=0)

        else:
            with torch.no_grad():
                rise_map = rise_mod(data).detach().cpu()

            if allRise is None:
                allRise = rise_map
            else:
                allRise = torch.cat((allRise,rise_map),dim=0)

    suff = "" if args.viz_id == "" else "{}_".format(args.viz_id)
    if not args.rise and not args.score_map and not args.noise_tunnel:
        np.save("../results/{}/gradcam_{}_epoch{}_{}test.npy".format(args.exp_id,args.model_id,bestEpoch,suff),allMask.numpy())
        np.save("../results/{}/gradcam_pp_{}_epoch{}_{}test.npy".format(args.exp_id,args.model_id,bestEpoch,suff),allMask_pp.numpy())
        np.save("../results/{}/gradcam_maps_{}_epoch{}_{}test.npy".format(args.exp_id,args.model_id,bestEpoch,suff),allMaps.numpy())
    elif args.score_map:
        np.save("../results/{}/score_maps_{}_epoch{}_{}test.npy".format(args.exp_id,args.model_id,bestEpoch,suff),allScore.numpy())
    elif args.noise_tunnel:
        np.save("../results/{}/smoothgrad_sq_{}_epoch{}_{}test.npy".format(args.exp_id,args.model_id,bestEpoch,suff),allSq.numpy())
        np.save("../results/{}/vargrad_{}_epoch{}_{}test.npy".format(args.exp_id,args.model_id,bestEpoch,suff),allVar.numpy())
    else:
        res_str = '' if args.rise_resolution == 7 else str(args.rise_resolution)
        np.save("../results/{}/rise{}_maps_{}_epoch{}_{}test.npy".format(args.exp_id,res_str,args.model_id,bestEpoch,suff),allRise.numpy())

if __name__ == "__main__":
    main()