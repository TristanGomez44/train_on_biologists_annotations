import numpy as np 
import scipy.stats
import matplotlib.pyplot as plt 
import torchvision 
import torch
import os
import args
import utils
from args import ArgReader
from args import str2bool

def main(argv=None):
    # Getting arguments from config file and command line
    # Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--transf1', type=str,default="identity")
    argreader.parser.add_argument('--transf2', type=str,default="identity")
    argreader.parser.add_argument('--class_map', action="store_true")
    
    argreader.parser.add_argument('--model_id2', type=str)

    # Reading the comand line arg
    argreader.getRemainingArgs()

    args = argreader.args

    if args.model_id2 is None:
        args.model_id2 = args.model_id

    repr1 = np.load(f"../results/{args.exp_id}/img_repr_model{args.model_id}_transf{args.transf1}.npy")
    repr2 = np.load(f"../results/{args.exp_id}/img_repr_model{args.model_id2}_transf{args.transf2}.npy")

    if args.model_id == args.model_id2:
        model_ids = args.model_id 
    else:
        model_ids = args.model_id + "_and_" + args.model_id2

    #################### Norm study ###########################

    norm1,norm2 = np.abs(repr1).sum(axis=1),np.abs(repr2).sum(axis=1)

    os.makedirs(f"../vis/{args.exp_id}/repr_attMap/",exist_ok=True)

    xmin,xmax = 200,2000
    plt.figure()
    plt.hist(norm1,label=args.transf1,alpha=0.5,range=(xmin,xmax),bins=20)
    plt.hist(norm2,label=args.transf2,alpha=0.5,range=(xmin,xmax),bins=20)
    plt.xlim(xmin,xmax)
    plt.ylim(0,30)
    plt.legend()
    plt.savefig(f"../vis/{args.exp_id}/repr_attMap/{args.transf1}_vs_{args.transf2}_repre_distr_{model_ids}.png")
    plt.close()

    stat,pvalue = scipy.stats.ttest_ind(norm1,norm2,equal_var=False)

    print(stat,pvalue)

    #################### Att maps ##########################

    if args.class_map:
        keyword = "classMap"
        color_map = plt.get_cmap('rainbow')
    else:
        keyword = "attMaps"
        color_map = plt.get_cmap('plasma')

    maps1 = np.load(f"../results/{args.exp_id}/img_{keyword}_model{args.model_id}_transf{args.transf1}.npy")
    maps2 = np.load(f"../results/{args.exp_id}/img_{keyword}_model{args.model_id2}_transf{args.transf2}.npy")

    imgs1 = torch.from_numpy(np.load(f"../results/{args.exp_id}/img_model{args.model_id}_transf{args.transf1}.npy"))
    imgs2 = torch.from_numpy(np.load(f"../results/{args.exp_id}/img_model{args.model_id2}_transf{args.transf2}.npy"))

    for i in range(min(len(maps1),10)):

        grid = []

        for maps,imgs in [[maps1,imgs1],[maps2,imgs2]]:
            viz_map = maps[i:i+1]

            if args.class_map:
                viz_map_raw = torch.from_numpy(viz_map)
                viz_map = viz_map_raw.argmax(dim=1,keepdim=True)

            viz_map = viz_map/max(maps1[i:i+1].max(),maps2[i:i+1].max())

            viz_mapShape = viz_map.shape
            viz_map = color_map(viz_map.reshape(-1))[:,:3].reshape((1,viz_mapShape[2],viz_mapShape[3],3))
            viz_map = torch.from_numpy(viz_map).permute(0,3,1,2)

            if args.class_map:
                viz_map_pred = viz_map_raw.max(dim=1,keepdim=True)[0]
                viz_map_pred_norm = viz_map_pred/viz_map_pred.max()
                viz_map = viz_map * viz_map_pred_norm

            img = imgs[i:i+1]
            img = utils.inv_imgnet_norm(img)*(img!=0)

            viz_map = torch.nn.functional.interpolate(viz_map,(img.shape[2],img.shape[3]),mode="nearest")

            img_viz_map = 0.8*viz_map + 0.2*img

            grid.extend([img,img_viz_map])

        grid = torch.cat(grid,dim=0)

        torchvision.utils.save_image(grid,f"../vis/{args.exp_id}/repr_attMap/repr_{keyword}_analysis_{model_ids}_{i}_{args.transf1}_vs_{args.transf2}.png")

if __name__ == "__main__":
    main()

