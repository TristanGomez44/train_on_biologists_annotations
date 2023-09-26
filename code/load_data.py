import glob,os
import numpy as np
import torch
import args
from random import Random
import grade_dataset
import ssl_dataset
from torch.utils.data.sampler import Sampler

def get_class_nb(dataset_train):
    args.class_nb = len(glob.glob(os.path.join("../data/",dataset_train,"*/")))
    if args.class_nb == 0:
        raise ValueError("Found 0 classes in",os.path.join(dataset_train,"*/"))
    return args.class_nb 
    
def get_img_size(args):
    if "vit" in args.first_mod or "dit" in args.first_mod:
        imgSize = 224
    elif args.first_mod=="swin_b_16":
        imgSize = 192
    elif args.first_mod=="swin_b_8":
        imgSize = 224
    elif args.big_images:
        imgSize = 512,384
    else:
        imgSize = 256,192
    return imgSize

def buildTrainLoader(args,shuffle=True):

    imgSize = get_img_size(args)

    if args.ssl:
        train_dataset = ssl_dataset.SSLDataset(args.dataset_path,"train",args.train_prop,args.val_prop,imgSize,augment=args.ssl_data_augment)
    else:    
        train_dataset = grade_dataset.GradeDataset(args.dataset_name,args.dataset_path,"train",imgSize)

        np.random.seed(1)
        torch.manual_seed(1)
        if args.cuda:
            torch.cuda.manual_seed(1)
        
    bsz = args.batch_size
    bsz = bsz if bsz < args.max_batch_size_single_pass else args.max_batch_size_single_pass

    kwargs = {"shuffle": shuffle}

    trainLoader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bsz,
                                              pin_memory=True, num_workers=args.num_workers,drop_last=args.drop_last, **kwargs)

    return trainLoader, train_dataset

def buildTestLoader(args, mode):
    imgSize = get_img_size(args)
    
    if args.ssl:
        test_dataset = ssl_dataset.SSLDataset(args.dataset_path,mode,args.train_prop,args.val_prop,imgSize,augment=args.ssl_data_augment)
    else:
        test_dataset = grade_dataset.GradeDataset(args.dataset_name,args.dataset_path,mode,imgSize)

    if args.val_batch_size == -1:
        args.val_batch_size = int(args.max_batch_size*3.5)

    testLoader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.val_batch_size,
                                             num_workers=args.num_workers,pin_memory=True)

    return testLoader,test_dataset

def addArgs(argreader):
    argreader.parser.add_argument('--pretrain_dataset', type=str, metavar='N',
                                  help='The network producing the features can only be pretrained on \'imageNet\'. This argument must be \
                            set to \'imageNet\' datasets.')
    argreader.parser.add_argument('--batch_size', type=int, metavar='BS',
                                  help='The batchsize to use for training')
    argreader.parser.add_argument('--val_batch_size', type=int, metavar='BS',
                                  help='The batchsize to use for validation')
    argreader.parser.add_argument('--max_batch_size_single_pass', type=int, metavar='BS',
                                  help='The maximum sub batch size when using very big images.')

    argreader.parser.add_argument('--train_prop', type=float, metavar='END',
                                  help='The proportion of the dataset to use for training')
    argreader.parser.add_argument('--val_prop', type=float, metavar='END',
                                  help='The proportion of the dataset to use for validation')

    argreader.parser.add_argument('--dataset_path', type=str, metavar='N')

    argreader.parser.add_argument('--dataset_name', type=str, metavar='N')
                              
    argreader.parser.add_argument('--shuffle_test_set', type=args.str2bool, metavar='BOOL',
                                  help='To shuffle the test set.')

    argreader.parser.add_argument('--icm_te_class_nb', type=int, metavar='S',
                                  help='The number of class for the ICM and TE')
    argreader.parser.add_argument('--exp_class_nb', type=int, metavar='S',help='The number of class for the blastocyst expansion.')

    argreader.parser.add_argument('--old_preprocess', type=args.str2bool, metavar='S',
                                  help='To use the old images pre-processor.')
    argreader.parser.add_argument('--moredataaug_preprocess', type=args.str2bool, metavar='S',
                                  help='To apply color jitter and random rotation along random resized crop and horizontal flip')
    argreader.parser.add_argument('--ws_dan_preprocess', type=args.str2bool, metavar='S',
                                  help='To apply the same image preprocessing as in WS-DAN.')

    argreader.parser.add_argument('--upscale_test', type=args.str2bool, metavar='S',
                                  help='To increase test resolution from 224 to 312')

    argreader.parser.add_argument('--big_images', type=args.str2bool, metavar='S',
                                  help='To resize the images to 448 pixels instead of 224')
    argreader.parser.add_argument('--very_big_images', type=args.str2bool, metavar='S',
                                    help='To resize the images to 1792 pixels instead of 224')


    argreader.parser.add_argument('--drop_last', type=args.str2bool, metavar='S',
                                    help='To drop the last batch in case it does not fit the batch size.')


    argreader.parser.add_argument('--normalize_data', type=args.str2bool, metavar='S',
                                  help='To normalize the data using imagenet means and std before puting it between 0 and 1.')

    argreader.parser.add_argument('--with_seg', type=args.str2bool, metavar='S',
                                  help='To load segmentation along with image and target')

    argreader.parser.add_argument('--repr_vec', type=args.str2bool, metavar='S',
                                  help='To use representative vectors instead of raw image.')

    argreader.parser.add_argument('--sq_resizing', type=args.str2bool, metavar='S',
                                  help='To resize each input image to a square.')

    argreader.parser.add_argument('--crop_ratio', type=float, metavar='S',
                                  help='The crop ratio (usually 0.875) for data augmentation.')
    argreader.parser.add_argument('--brightness', type=float, metavar='S',
                                  help='The brightness intensity for data augmentation.')
    argreader.parser.add_argument('--saturation', type=float, metavar='S',
                                  help='The saturation intensity for data augmentation.')

    argreader.parser.add_argument('--add_patches', type=args.str2bool, metavar='S',
                                  help='To add black patches during data augmentation.')

    argreader.parser.add_argument('--patch_res', type=int, metavar='S',
                                  help='The resolution of the black patch mask')

    return argreader
