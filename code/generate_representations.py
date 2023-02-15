import glob
import numpy as np 
import torch 

from args import addInitArgs,addValArgs
from init_model import preprocessAndLoadParams
from compute_scores_for_saliency_metrics import getBatch,init_post_hoc_arg,sample_img_inds,get_other_img_inds
import modelBuilder,load_data
import sparse
from args import ArgReader
from sal_metr_data_aug import apply_sal_metr_masks

MIN_PATCH_SIZE = 10 
MAX_PATCH_SIZE = 100

PATCH_NB = 10

def apply_transf(data,transf,data_bckgr,sparse_conv=False,downsample_ratio=None,model=None):
    if transf == "identity":
        return data
    elif transf == "saliency_metrics":
        data,_ = apply_sal_metr_masks(model,data)
        return data
    else:
        
        if transf.find("nb") != -1:
            nb_patches = int(transf.split("nb")[1].split("_")[0])
        else:
            nb_patches = PATCH_NB

        if transf.find("size") != -1:
            min_size = int(transf.split("size")[1].split("_")[0])
            max_size = min_size + 1
        else:
            min_size = MIN_PATCH_SIZE 
            max_size = MAX_PATCH_SIZE

        xmax,ymax = data.shape[2],data.shape[3]
        if sparse_conv:
            xmax = xmax//downsample_ratio
            ymax = ymax//downsample_ratio

        sizes = np.random.randint(min_size,max_size,size=(nb_patches))
        x = np.random.randint(0,xmax-1,size=(nb_patches))
        y = np.random.randint(0,ymax-1,size=(nb_patches))

        mask = torch.ones(1,1,xmax,ymax)

        for i in range(nb_patches):
            mask[:,:,x[i]:x[i]+sizes[i],y[i]:y[i]+sizes[i]] = 0

        if transf.find("blur") != -1:
            kernel = torch.ones(21,21)
            kernel = kernel/kernel.numel()
            kernel = kernel.unsqueeze(0).unsqueeze(0).expand(3,1,-1,-1)
            mask = torch.nn.functional.conv2d(mask,kernel,padding=kernel.size(-1)//2,groups=kernel.size(0))

        if sparse_conv:
            sparse._cur_active = mask
            mask_imgsize = torch.nn.functional.interpolate(mask,data.shape[2:],mode="nearest")
        else:
            mask_imgsize = mask

        if transf.find("black_patches") != -1:
            data = data*mask_imgsize
        elif transf.find("img") != -1:
            data = data*mask_imgsize + data_bckgr*(1-mask_imgsize)
        else:
            raise ValueError("Unkown transformation function",transf)

        return data 

def main(argv=None):
    # Getting arguments from config file and command line
    # Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--transf', type=str,default="identity")
    argreader.parser.add_argument('--att_metrics_img_nb', type=int, help='The nb of images on which to compute the att metric.')
   
    argreader.parser.add_argument('--class_map',action="store_true")
    argreader.parser.add_argument('--sparse',action="store_true")

    argreader = addInitArgs(argreader)
    argreader = addValArgs(argreader)
    argreader = init_post_hoc_arg(argreader)

    argreader = modelBuilder.addArgs(argreader)
    argreader = load_data.addArgs(argreader)

    # Reading the comand line arg
    argreader.getRemainingArgs()

    args = argreader.args

    args.cuda = args.cuda and torch.cuda.is_available()

    bestPath = glob.glob("../models/{}/model{}_best_epoch*".format(args.exp_id, args.model_id))[0]
    net = modelBuilder.netBuilder(args,gpu=0)
    net = preprocessAndLoadParams(bestPath,args.cuda,net)
    
    if args.sparse:
        imgSize = load_data.get_img_size(args)
        downsample_ratio = modelBuilder.getResnetDownSampleRatio(args.first_mod,args)
        fea_nb = modelBuilder.getResnetFeat(args.first_mod, args.resnet_chan)
        net = sparse.SparseEncoder(net, imgSize, downsample_ratio, fea_nb)
        net.secondModel = net.sp_cnn.secondModel
    else:
        downsample_ratio = None

    net.eval()
    
    args.val_batch_size = 1
    _,testDataset = load_data.buildTestLoader(args, "test")

    np.random.seed(0)
    inds = sample_img_inds(testDataset,args.img_nb_per_class)

    if args.class_map:
        inds = inds[:10]

    if args.transf.find("img") != -1:
        inds_bckgr = get_other_img_inds(inds)
    else:
        inds_bckgr = None

    torch.random.manual_seed(0)

    featList = []
    attMapList = []
    imgList = []
    predMapList = []
    with torch.no_grad():
        for i in range(len(inds)):
            if i % 20 == 0 :
                print("Img",inds[i].item(),"(",i,"/",len(inds),")")

            data,_ = getBatch(testDataset,[inds[i]],args)

            if args.transf.find("img") != -1:
                data_bckgr,_ = getBatch(testDataset,[inds_bckgr[i]],args)
            else:
                data_bckgr = None

            data_transf = apply_transf(data,args.transf,data_bckgr,args.sparse,downsample_ratio,net)

            resDic = net(data_transf)

            featList.append(resDic["feat_pooled"])
            attMapList.append(torch.abs(resDic["feat"]).sum(dim=1,keepdim=True))
            imgList.append(data_transf)

            if args.class_map:
                feat = resDic["feat"]
                shape = feat.shape
                feat = feat.view(shape[0],shape[1],-1)
                feat = feat.permute(0,2,1)
                feat = feat.reshape(shape[0]*shape[2]*shape[3],-1)

                pred = net.secondModel({"x":feat})["pred"]
                shape = pred.shape
                pred = pred.permute(1,0)
                map_size = int(np.sqrt(shape[0]))
                pred = pred.view(1,shape[1],map_size,map_size)

                predMapList.append(pred)

    featList = torch.cat(featList,axis=0).cpu().numpy()
    attMapList = torch.cat(attMapList,axis=0).cpu().numpy()
    imgList = torch.cat(imgList,dim=0).cpu().numpy()
 
    model_id = args.model_id
    if args.sparse:
        model_id += "_spars"

    if not args.class_map:
        np.save(f"../results/{args.exp_id}/img_repr_model{model_id}_transf{args.transf}.npy",featList)
        np.save(f"../results/{args.exp_id}/img_attMaps_model{model_id}_transf{args.transf}.npy",attMapList)
        np.save(f"../results/{args.exp_id}/img_model{model_id}_transf{args.transf}.npy",imgList)

    if args.class_map:
        predMapList = torch.cat(predMapList,dim=0).cpu().numpy()
        np.save(f"../results/{args.exp_id}/img_classMap_model{model_id}_transf{args.transf}.npy",predMapList)
        
if __name__ == "__main__":
    main()