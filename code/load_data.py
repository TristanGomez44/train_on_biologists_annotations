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
import albumentations
import scipy.io
import imageDatasetWithSeg

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



def buildTrainLoader(args,transf=None,shuffle=True,withSeg=False):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    resizedImgSize = 500 if args.big_images else 224
    if transf is None:

        if args.old_preprocess:
            transf = transforms.Compose(
                [transforms.RandomResizedCrop(resizedImgSize), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
        elif args.moredataaug_preprocess:

            albTransfFunc = albumentations.Compose([
                albumentations.augmentations.transforms.GaussNoise(var_limit=(10.0, 100.0)),
                albumentations.augmentations.transforms.GaussianBlur(blur_limit=10)])

            transf = transforms.Compose(
                [transforms.RandomResizedCrop(resizedImgSize, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(),
                torchvision.transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                torchvision.transforms.Lambda(lambda x:albTransfFunc(image=np.asarray(x))["image"]),
                transforms.ToTensor()])
        else:
            transf = transforms.Compose([transforms.Resize(resizedImgSize), transforms.RandomCrop(resizedImgSize, padding=0, pad_if_needed=True),
                                         transforms.RandomHorizontalFlip(), transforms.ToTensor()])

        if args.normalize_data:
            transf = transforms.Compose([transf,normalize])

    if transf == "identity":
        transf = transforms.Compose([transforms.Resize((resizedImgSize,resizedImgSize)), transforms.ToTensor()])

    if withSeg:
        datasetConst = imageDatasetWithSeg.ImageFolderWithSeg
    else:
        datasetConst = torchvision.datasets.ImageFolder

    train_dataset = datasetConst("../data/{}".format(args.dataset_train), transf)

    totalLength = len(train_dataset)

    if args.prop_set_int_fmt:
        train_prop = args.train_prop / 100
    else:
        train_prop = args.train_prop

    np.random.seed(1)
    torch.manual_seed(1)
    if args.cuda:
        torch.cuda.manual_seed(1)

    train_dataset, _ = torch.utils.data.random_split(train_dataset, [int(totalLength * train_prop),
                                                                     totalLength - int(totalLength * train_prop)])

    kwargs = {"shuffle": shuffle}

    if args.distributed:
        size = dist.get_world_size()
        bsz = int(args.batch_size / float(size))
        partition_sizes = [1.0 / size for _ in range(size)]
        partition = DataPartitioner(train_dataset, partition_sizes)
        partition = partition.use(dist.get_rank())
    else:
        bsz = args.batch_size

    trainLoader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=bsz,  # use custom collate function here
                                              pin_memory=False, num_workers=args.num_workers, **kwargs)

    return trainLoader, train_dataset


def buildTestLoader(args, mode,shuffle=False,withSeg=False):
    datasetName = getattr(args, "dataset_{}".format(mode))

    resizedImgSize = 500 if args.big_images else 224

    if args.old_preprocess:
        transf = transforms.Compose([transforms.Resize(int(resizedImgSize*1.14)), transforms.CenterCrop(resizedImgSize), transforms.ToTensor()])
    else:
        transf = transforms.Compose([transforms.Resize(resizedImgSize), transforms.CenterCrop(resizedImgSize), transforms.ToTensor()])

    if args.normalize_data:
        normalizeFunc = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transf = transforms.Compose([transf, normalizeFunc])

    if withSeg:
        datasetConst = imageDatasetWithSeg.ImageFolderWithSeg
    else:
        datasetConst = torchvision.datasets.ImageFolder

    test_dataset = datasetConst("../data/{}".format(datasetName), transf)

    if mode == "val" and args.dataset_train == args.dataset_val:
        np.random.seed(1)
        torch.manual_seed(1)
        if args.cuda:
            torch.cuda.manual_seed(1)

        if args.prop_set_int_fmt:
            train_prop = args.train_prop / 100
        else:
            train_prop = args.train_prop

        totalLength = len(test_dataset)
        _, test_dataset = torch.utils.data.random_split(test_dataset, [int(totalLength * train_prop),
                                                                       totalLength - int(totalLength * train_prop)])

        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if args.cuda:
            torch.cuda.manual_seed(args.seed)

    testLoader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=args.val_batch_size,
                                             num_workers=args.num_workers,shuffle=shuffle)

    return testLoader,test_dataset


def addArgs(argreader):
    argreader.parser.add_argument('--pretrain_dataset', type=str, metavar='N',
                                  help='The network producing the features can only be pretrained on \'imageNet\'. This argument must be \
                            set to \'imageNet\' datasets.')
    argreader.parser.add_argument('--batch_size', type=int, metavar='BS',
                                  help='The batchsize to use for training')
    argreader.parser.add_argument('--val_batch_size', type=int, metavar='BS',
                                  help='The batchsize to use for validation')

    argreader.parser.add_argument('--train_prop', type=float, metavar='END',
                                  help='The proportion of the train dataset to use for training when working in non video mode. The rest will be used for validation.')

    argreader.parser.add_argument('--prop_set_int_fmt', type=args.str2bool, metavar='BOOL',
                                  help='Set to True to set the sets (train, validation and test) proportions\
                            using int between 0 and 100 instead of float between 0 and 1.')

    argreader.parser.add_argument('--dataset_train', type=str, metavar='DATASET',
                                  help='The dataset for training. Can be "big" or "small"')
    argreader.parser.add_argument('--dataset_val', type=str, metavar='DATASET',
                                  help='The dataset for validation. Can be "big" or "small"')
    argreader.parser.add_argument('--dataset_test', type=str, metavar='DATASET',
                                  help='The dataset for testing. Can be "big" or "small"')

    argreader.parser.add_argument('--class_nb', type=int, metavar='S',
                                  help='The number of class of to model')

    argreader.parser.add_argument('--old_preprocess', type=args.str2bool, metavar='S',
                                  help='To use the old images pre-processor.')
    argreader.parser.add_argument('--moredataaug_preprocess', type=args.str2bool, metavar='S',
                                  help='To apply color jitter and random rotation along random resized crop and horizontal flip')

    argreader.parser.add_argument('--big_images', type=args.str2bool, metavar='S',
                                  help='To resize the images to 500 pixels instead of 224')
    argreader.parser.add_argument('--normalize_data', type=args.str2bool, metavar='S',
                                  help='To normalize the data using imagenet means and std before puting it between 0 and 1.')

    argreader.parser.add_argument('--with_seg', type=args.str2bool, metavar='S',
                                  help='To load segmentation along with image and target')


    return argreader
