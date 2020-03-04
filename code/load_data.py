import sys
import glob
import os

import numpy as np
import torch
from torchvision import transforms
import torchvision

from PIL import Image

import time
import args
import utils
import torch.distributed as dist
from random import Random

class Partition(object):

    def __init__(self, data, index):
        self.data = data
        self.index = index

    def __len__(self):
        return len(self.index)

    def __getitem__(self, index):
        data_idx = self.index[index]
        return self.data[data_idx]

class DataPartitioner(object):

    def __init__(self, data, sizes=[0.7, 0.2, 0.1], seed=1234):
        self.data = data
        self.partitions = []
        rng = Random()
        rng.seed(seed)
        data_len = len(data)
        indexes = [x for x in range(0, data_len)]
        rng.shuffle(indexes)

        for frac in sizes:
            part_len = int(frac * data_len)
            self.partitions.append(indexes[0:part_len])
            indexes = indexes[part_len:]

    def use(self, partition):
        return Partition(self.data, self.partitions[partition])

def buildTrainLoader(args):

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    transf = transforms.Compose([transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),normalize])

    train_dataset = torchvision.datasets.ImageFolder("../data/{}".format(args.dataset_train),transf)
    totalLength = len(train_dataset)

    if args.prop_set_int_fmt:
        train_prop = args.train_prop/100
    else:
        train_prop = args.train_prop

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)

    train_dataset,_ = torch.utils.data.random_split(train_dataset, [int(totalLength*train_prop),totalLength-int(totalLength*train_prop)])

    kwargs = {"shuffle":True}

    if args.distributed:
        size = dist.get_world_size()
        bsz = int(args.batch_size / float(size))
        partition_sizes = [1.0 / size for _ in range(size)]
        partition = DataPartitioner(train_dataset, partition_sizes)
        partition = partition.use(dist.get_rank())
    else:
        bsz = args.batch_size

    trainLoader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=bsz, # use custom collate function here
                      pin_memory=False,num_workers=args.num_workers,**kwargs)

    return trainLoader,train_dataset

def buildTestLoader(args,mode,normalize=True):

    datasetName = getattr(args,"dataset_{}".format(mode))

    if normalize:
        normalizeFunc = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        transf = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),normalizeFunc])
    else:
        transf = transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor()])

    test_dataset = torchvision.datasets.ImageFolder("../data/{}".format(datasetName),transf)

    if mode == "val" and args.dataset_train == args.dataset_val:
        np.random.seed(1)
        torch.manual_seed(1)
        if args.cuda:
            torch.cuda.manual_seed(1)

        if args.prop_set_int_fmt:
            train_prop = args.train_prop/100
        else:
            train_prop = args.train_prop

        totalLength = len(test_dataset)
        _,test_dataset = torch.utils.data.random_split(test_dataset, [int(totalLength*train_prop),totalLength-int(totalLength*train_prop)])

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

    testLoader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=args.val_batch_size,num_workers=args.num_workers)

    return testLoader

def addArgs(argreader):

    argreader.parser.add_argument('--pretrain_dataset', type=str, metavar='N',
                        help='The network producing the features can only be pretrained on \'imageNet\'. This argument must be \
                            set to \'imageNet\' datasets.')
    argreader.parser.add_argument('--batch_size', type=int,metavar='BS',
                        help='The batchsize to use for training')
    argreader.parser.add_argument('--val_batch_size', type=int,metavar='BS',
                        help='The batchsize to use for validation')

    argreader.parser.add_argument('--tr_len', type=int,metavar='LMAX',
                        help='The maximum length of a training sequence')
    argreader.parser.add_argument('--val_l', type=int,metavar='LMAX',
                        help='Length of sequences for validation.')

    argreader.parser.add_argument('--img_size', type=int,metavar='EDGE_SIZE',
                        help='The size of each edge of the images after resizing, if resizing is desired. Else is should be equal to --orig_img_size')
    argreader.parser.add_argument('--orig_img_size', type=int,metavar='EDGE_SIZE',
                        help='The size of each edge of the images before preprocessing.')

    argreader.parser.add_argument('--train_prop', type=float,metavar='END',
                        help='The proportion of the train dataset to use for training when working in non video mode. The rest will be used for validation.')

    argreader.parser.add_argument('--prop_set_int_fmt', type=args.str2bool,metavar='BOOL',
                        help='Set to True to set the sets (train, validation and test) proportions\
                            using int between 0 and 100 instead of float between 0 and 1.')


    argreader.parser.add_argument('--dataset_train', type=str,metavar='DATASET',
                        help='The dataset for training. Can be "big" or "small"')
    argreader.parser.add_argument('--dataset_val', type=str,metavar='DATASET',
                        help='The dataset for validation. Can be "big" or "small"')
    argreader.parser.add_argument('--dataset_test', type=str,metavar='DATASET',
                        help='The dataset for testing. Can be "big" or "small"')

    argreader.parser.add_argument('--resize_image', type=args.str2bool, metavar='S',
                        help='to resize the image to the size indicated by the img_width and img_heigth arguments.')

    argreader.parser.add_argument('--class_nb', type=int, metavar='S',
                        help='The number of class of to model')

    argreader.parser.add_argument('--augment_data', type=args.str2bool, metavar='S',
                        help='Set to True to augment the training data with transformations')

    argreader.parser.add_argument('--mask_time_on_image', type=args.str2bool, metavar='S',
                        help='To mask the time displayed on the images')

    argreader.parser.add_argument('--min_phase_nb', type=int, metavar='S',
                        help='The minimum number of phases a video must have to be included in the dataset')

    argreader.parser.add_argument('--grid_shuffle', type=args.str2bool, metavar='S',
                        help='Apply a grid shuffle transformation from albumentation to the training images')

    argreader.parser.add_argument('--grid_shuffle_size', type=int, metavar='S',
                        help='The grid size for grid shuffle.')


    argreader.parser.add_argument('--grid_shuffle_test', type=args.str2bool, metavar='S',
                        help='Apply a grid shuffle transformation from albumentation to the testing images')

    argreader.parser.add_argument('--grid_shuffle_test_size', type=int, metavar='S',
                        help='The grid size for grid shuffle for the test phase.')

    argreader.parser.add_argument('--sobel', type=args.str2bool, metavar='S',
                        help='To apply sobel transform to each image')


    return argreader
