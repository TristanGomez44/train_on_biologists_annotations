import sys
import glob
import os

import numpy as np
import torch
from torchvision import transforms
import torchvision

from PIL import Image

import processResults
import pims
import time

import args
import warnings
warnings.filterwarnings('ignore',module=".*av.*")

import utils
import formatData

import albumentations
from albumentations import Compose

import digitExtractor
import cv2

class Sampler(torch.utils.data.sampler.Sampler):
    """ The sampler for the SeqTrDataset dataset
    """

    def __init__(self, nb_videos,nb_images,seqLen):
        self.nb_videos = nb_videos
        self.nb_images = nb_images
        self.seqLen = seqLen
    def __iter__(self):

        if self.nb_images > 0:
            return iter(torch.randint(0,self.nb_videos,size=(self.nb_images//self.seqLen,)))
        else:
            return iter([])

    def __len__(self):
        return self.nb_images

def collateSeq(batch):

    res = list(zip(*batch))

    res[0] = torch.cat(res[0],dim=0)
    if not res[1][0] is None:
        res[1] = torch.cat(res[1],dim=0)

    if torch.is_tensor(res[2][0]):
        res[2] = torch.cat(res[2],dim=0)

    return res

class SeqTrDataset(torch.utils.data.Dataset):
    '''
    The dataset to sample sequence of frames from videos

    When the method __getitem__(i) is called, the dataset randomly select a sequence from the video i

    Args:
    - propStart (float): the proportion of the dataset at which to start using the videos. For example : propEnd=0.5 and propEnd=1 will only use the last half of the videos
    - propEnd (float): the proportion of the dataset at which to stop using the videos. For example : propEnd=0 and propEnd=0.5 will only use the first half of the videos
    - trLen (int): the length of a sequence during training
    - imgSize (int): the size of each side of the image
    - resizeImage (bool): a boolean to indicate if the image should be resized using cropping or not
    - exp_id (str): the name of the experience
    '''

    def __init__(self,dataset,propStart,propEnd,propSetIntFormat,trLen,imgSize,origImgSize,resizeImage,exp_id,augmentData,maskTime):

        super(SeqTrDataset, self).__init__()

        self.dataset = dataset
        self.videoPaths = findVideos(dataset,propStart,propEnd,propSetIntFormat)

        print("Number of training videos : ",len(self.videoPaths))
        self.imgSize = imgSize
        self.trLen = trLen
        self.nbImages = 0
        self.exp_id = exp_id

        if propStart != propEnd:
            for videoPath in self.videoPaths:
                fps = utils.getVideoFPS(videoPath)
                self.nbImages += utils.getVideoFrameNb(videoPath)

        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.toTensor = torchvision.transforms.ToTensor()
        self.resizeImage = resizeImage

        if self.resizeImage:
            self.reSizeFunc = torchvision.transforms.Compose([torchvision.transforms.ToPILImage(),torchvision.transforms.Resize(imgSize)])
        else:
            self.reSizeFunc = None

        self.FPSDict = {}

        self.augmentData = augmentData
        if augmentData:
            self.transf = Compose([
                    albumentations.RandomBrightness(limit=0.2, p=0.5),
                    albumentations.ElasticTransform(alpha=25, sigma=25, alpha_affine=25, p=0.5),
                    albumentations.OpticalDistortion(distort_limit=0.1, shift_limit=0.1, p=0.5),
                    albumentations.Flip(p=0.5),
                    albumentations.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=360, p=0.5),
                    albumentations.RandomSizedCrop((int(0.736*imgSize),imgSize), imgSize, imgSize, p=0.5),
                    albumentations.RandomContrast(limit=0, p=0.5)
                ], p=1)
        else:
            self.transf = None

        self.maskTime = maskTime
        self.mask = computeMask(maskTime,origImgSize)

    def __len__(self):
        return self.nbImages

    def __getitem__(self,vidInd):

        targ = torch.zeros(self.trLen)
        vidNames = []

        vidName = os.path.basename(os.path.splitext(self.videoPaths[vidInd])[0])

        if not self.videoPaths[vidInd] in self.FPSDict.keys():
            self.FPSDict[self.videoPaths[vidInd]] = utils.getVideoFPS(self.videoPaths[vidInd])

        frameNb = utils.getVideoFrameNb(self.videoPaths[vidInd])

        #Computes the label index of each frame
        gt = getGT(vidName,self.dataset)
        frameInds = np.arange(frameNb)

        ################# Frame selection ##################
        startFrame = torch.randint(0,frameNb-self.trLen,size=(1,))
        frameInds,gt = frameInds[startFrame:startFrame+self.trLen],gt[startFrame:startFrame+self.trLen]

        video = pims.Video(self.videoPaths[vidInd])

        def preproc(x):

            x = video[x]
            if self.maskTime:
                x = x*self.mask[:,:,np.newaxis]

            if self.resizeImage:
                x = np.asarray(self.reSizeFunc(x))

            return x[np.newaxis,:,:,0]

        #Building the frame sequence
        #The videos are in black and white but there as still encoded using 3 channels
        #Therefore, the three channels carry the same values
        frameSeq = np.concatenate(list(map(preproc,np.array(frameInds))),axis=0)
        # Shape of tensor : T x H x W
        frameSeq = frameSeq.transpose((1,2,0))
        # H x W x T
        if self.augmentData:
            frameSeq = self.transf(image=frameSeq)["image"]
        # H x W x T
        frameSeq = self.toTensor(frameSeq)
        # T x H x W
        frameSeq = frameSeq.unsqueeze(1)
        # T x 1 x H x W
        frameSeq = frameSeq.expand(frameSeq.size(0),3,frameSeq.size(2),frameSeq.size(3))
        # T x 3 x H x W
        frameSeq = torch.cat(list(map(lambda x:self.normalize(x).unsqueeze(0),frameSeq.float())),dim=0)

        return frameSeq.unsqueeze(0),torch.tensor(gt).unsqueeze(0),vidName,torch.tensor(frameInds).int()

class TestLoader():
    '''
    The dataset to sample sequence of frames from videos. As the video contains a great number of frame,
    each video is processed through several batches and each batch contain only one sequence.

    Args:
    - evalL (int): the length of a sequence in a batch. A big value will reduce the number of batches necessary to process a whole video
    - dataset (str): the name of the dataset
    - propStart (float): the proportion of the dataset at which to start using the videos. For example : propEnd=0.5 and propEnd=1 will only use the last half of the videos
    - propEnd (float): the proportion of the dataset at which to stop using the videos. For example : propEnd=0 and propEnd=0.5 will only use the first half of the videos
    - imgSize (tuple): a tuple containing (in order) the width and size of the image
    - resizeImage (bool): a boolean to indicate if the image should be resized using cropping or not
    - exp_id (str): the name of the experience
    '''

    def __init__(self,dataset,evalL,propStart,propEnd,propSetIntFormat,imgSize,origImgSize,resizeImage,exp_id,maskTime):
        self.dataset = dataset
        self.evalL = evalL
        self.videoPaths = findVideos(dataset,propStart,propEnd,propSetIntFormat)
        self.exp_id = exp_id
        print("Number of eval videos",len(self.videoPaths))
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        transfList = []

        self.maskTime = maskTime
        self.mask = computeMask(maskTime,origImgSize)

        if maskTime:
            maskTrans = torchvision.transforms.Lambda(lambda x : x*self.mask[:,:,np.newaxis])
            transfList.append(maskTrans)

        if resizeImage:
            resizeTransf = transforms.Compose([torchvision.transforms.ToPILImage(),torchvision.transforms.Resize(imgSize)])
            transfList.append(resizeTransf)

        self.preproc = transforms.Compose(transfList+[transforms.ToTensor(),normalize])

        self.nbImages = 0
        for videoPath in self.videoPaths:
            fps = utils.getVideoFPS(videoPath)
            self.nbImages += utils.getVideoFrameNb(videoPath)

    def __iter__(self):
        self.videoInd = 0
        self.currFrameInd = 0
        self.sumL = 0
        return self

    def __next__(self):

        if self.videoInd == len(self.videoPaths):
            raise StopIteration

        L = self.evalL
        self.sumL += L

        videoPath = self.videoPaths[self.videoInd]
        video = pims.Video(videoPath)

        vidName = os.path.basename(os.path.splitext(videoPath)[0])

        fps = utils.getVideoFPS(videoPath)
        frameNb = utils.getVideoFrameNb(videoPath)

        frameInds = np.arange(self.currFrameInd,min(self.currFrameInd+L,frameNb))

        frameSeq = torch.cat(list(map(lambda x:self.preproc(video[x][:,:,0:1].repeat(repeats=3,axis=-1)).unsqueeze(0),np.array(frameInds))),dim=0)

        gt = getGT(vidName,self.dataset)[self.currFrameInd:min(self.currFrameInd+L,frameNb)]

        if frameInds[-1] + 1 == frameNb:
            self.currFrameInd = 0
            self.videoInd += 1
        else:
            self.currFrameInd += L

        return frameSeq.unsqueeze(0),torch.tensor(gt).unsqueeze(0),vidName,torch.tensor(frameInds).int()

def computeMask(maskTime,imgSize):
    if maskTime:
        mask = np.ones((imgSize,imgSize)).astype("uint8")
        Y1,Y2 = digitExtractor.getDigitYPos()
        mask[Y1-12:Y2+12] = 0
    else:
        mask = None
    return mask

def buildSeqTrainLoader(args):

    train_dataset = SeqTrDataset(args.dataset_train,args.train_part_beg,args.train_part_end,args.prop_set_int_fmt,args.tr_len,\
                                        args.img_size,args.orig_img_size,args.resize_image,args.exp_id,args.augment_data,args.mask_time)
    sampler = Sampler(len(train_dataset.videoPaths),train_dataset.nbImages,args.tr_len)
    trainLoader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=args.batch_size,sampler=sampler, collate_fn=collateSeq, # use custom collate function here
                      pin_memory=False,num_workers=args.num_workers)

    return trainLoader,train_dataset

def findVideos(dataset,propStart,propEnd,propSetIntFormat=False):

    allVideoPaths = sorted(glob.glob("../data/{}/*.avi".format(dataset)))

    if propSetIntFormat:
        propStart /= 100
        propEnd /= 100

    if propStart < propEnd:
        videoPaths = np.array(allVideoPaths)[int(propStart*len(allVideoPaths)):int(propEnd*len(allVideoPaths))]
    else:
        videoPaths = allVideoPaths[int(propStart*len(allVideoPaths)):]
        videoPaths += allVideoPaths[:int(propEnd*len(allVideoPaths))]
        videoPaths = np.array(videoPaths)

    return videoPaths

def getGT(vidName,dataset):
    ''' For one video, returns the label of each frame

    Args:
    - vidName (str): the video name. It is the name of the video file minus the extension.
    Returns:
    - gt (array): the list of labels corresponding to each image

    '''

    if not os.path.exists("../data/{}/annotations/{}_targ.csv".format(dataset,vidName)):

        phases = np.genfromtxt("../data/{}/annotations/{}_phases.csv".format(dataset,vidName),dtype=str,delimiter=",")

        gt = np.zeros((int(phases[-1,-1])+1))

        for phase in phases:
            #The dictionary called here convert the label into a integer

            gt[int(phase[1]):int(phase[2])+1] = formatData.getLabels()[phase[0]]

        np.savetxt("../data/{}/annotations/{}_targ.csv".format(dataset,vidName),gt)
    else:
        gt = np.genfromtxt("../data/{}/annotations/{}_targ.csv".format(dataset,vidName))

    return gt.astype(int)

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

    argreader.parser.add_argument('--train_part_beg', type=float,metavar='START',
                        help='The start position of the train set. If --prop_set_int_fmt is True, it should be int between 0 and 100, else it is a float between 0 and 1.')
    argreader.parser.add_argument('--train_part_end', type=float,metavar='END',
                        help='The end position of the train set. If --prop_set_int_fmt is True, it should be int between 0 and 100, else it is a float between 0 and 1.')
    argreader.parser.add_argument('--val_part_beg', type=float,metavar='START',
                        help='The start position of the validation set. If --prop_set_int_fmt is True, it should be int between 0 and 100, else it is a float between 0 and 1.')
    argreader.parser.add_argument('--val_part_end', type=float,metavar='END',
                        help='The end position of the validation set. If --prop_set_int_fmt is True, it should be int between 0 and 100, else it is a float between 0 and 1.')
    argreader.parser.add_argument('--test_part_beg', type=float,metavar='START',
                        help='The start position of the test set. If --prop_set_int_fmt is True, it should be int between 0 and 100, else it is a float between 0 and 1.')
    argreader.parser.add_argument('--test_part_end', type=float,metavar='END',
                        help='The end position of the test set. If --prop_set_int_fmt is True, it should be int between 0 and 100, else it is a float between 0 and 1.')

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

    argreader.parser.add_argument('--mask_time', type=args.str2bool, metavar='S',
                        help='To mask the time displayed on the images')

    return argreader

if __name__ == "__main__":

    train_part_beg = 0
    train_part_end = 0.5
    val_part_beg = 0.5
    val_part_end = 1
    dataset_train = "small"
    dataset_val = "small"

    tr_len = 5
    val_l = 5
    img_size = 224
    orig_img_size = 500
    resize_image = True
    exp_id = "Test"

    batch_size = 5
    num_workers = 1
    augmentData = True
    maskTime = True

    train_dataset = SeqTrDataset(dataset_train,train_part_beg,train_part_end,tr_len,\
                                        img_size,orig_img_size,resize_image,exp_id,augmentData,maskTime)
    sampler = Sampler(len(train_dataset.videoPaths),train_dataset.nbImages,tr_len)
    trainLoader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,sampler=sampler, collate_fn=collateSeq, # use custom collate function here
                      pin_memory=False,num_workers=num_workers)

    for batch in trainLoader:
        print(batch[0].shape,batch[1].shape,batch[2])
        break

    valLoader = TestLoader(dataset_val,val_l,val_part_beg,val_part_end,\
                                        img_size,orig_img_size,resize_image,\
                                        exp_id,maskTime)

    for batch in valLoader:
        print(batch[0].shape,batch[1].shape,batch[2],batch[3])
        break
