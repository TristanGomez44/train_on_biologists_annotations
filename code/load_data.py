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

class Sampler(torch.utils.data.sampler.Sampler):
    """ The sampler for the SeqTrDataset dataset
    """

    def __init__(self, nb_videos,nb_images):
        self.nb_videos = nb_videos
        self.nb_images = nb_images
    def __iter__(self):
        return iter(torch.randint(0,self.nb_videos,size=(self.nb_images,)))

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

    def __init__(self,propStart,propEnd,trLen,imgSize,resizeImage,exp_id):

        super(SeqTrDataset, self).__init__()

        self.videoPaths = findVideos(propStart,propEnd)

        self.videoPaths = np.array(self.videoPaths)[int(propStart*len(self.videoPaths)):int(propEnd*len(self.videoPaths))]
        self.imgSize = imgSize
        self.trLen = trLen
        self.nbImages = 0
        self.exp_id = exp_id

        if propStart != propEnd:
            for videoPath in self.videoPaths:
                fps = utils.getVideoFPS(videoPath)
                self.nbImages += utils.getVideoFrameNb(videoPath)

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.preproc = transforms.Compose([transforms.ToPILImage(),transforms.ToTensor(),normalize])
        self.FPSDict = {}

    def __len__(self):
        return self.nbImages

    def __getitem__(self,vidInd):

        data = torch.zeros(self.trLen,3,self.imgSize,self.imgSize)
        targ = torch.zeros(self.trLen)
        vidNames = []

        vidName = os.path.basename(os.path.splitext(self.videoPaths[vidInd])[0])

        if not self.videoPaths[vidInd] in self.FPSDict.keys():
            self.FPSDict[self.videoPaths[vidInd]] = utils.getVideoFPS(self.videoPaths[vidInd])

        frameNb = utils.getVideoFrameNb(self.videoPaths[vidInd])

        #Computes the label index of each frame
        gt = getGT(vidName)
        frameInds = np.arange(frameNb)

        ################# Frame selection ##################
        startFrame = np.random.randint(frameNb-self.trLen)
        frameInds,gt = frameInds[startFrame:startFrame+self.trLen],gt[startFrame:startFrame+self.trLen]

        video = pims.Video(self.videoPaths[vidInd])

        #Building the frame sequence
        frameSeq = torch.cat(list(map(lambda x:self.preproc(video[x]).unsqueeze(0),np.array(frameInds))),dim=0)

        return frameSeq.unsqueeze(0),torch.tensor(gt).unsqueeze(0),vidName

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

    def __init__(self,evalL,propStart,propEnd,imgSize,resizeImage,exp_id):
        self.evalL = evalL
        self.videoPaths = findVideos(propStart,propEnd)
        self.exp_id = exp_id

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.preproc = transforms.Compose([transforms.ToTensor(),normalize])

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
        frameSeq = torch.cat(list(map(lambda x:self.preproc(video[x]).unsqueeze(0),np.array(frameInds))),dim=0)

        gt = getGT(vidName)[self.currFrameInd:min(self.currFrameInd+L,frameNb)]

        if frameInds[-1] + 1 == frameNb:
            self.currFrameInd = 0
            self.videoInd += 1
        else:
            self.currFrameInd += L

        return frameSeq.unsqueeze(0),torch.tensor(gt).unsqueeze(0),vidName,torch.tensor(frameInds).int()

def buildSeqTrainLoader(args):

    train_dataset = SeqTrDataset(args.train_part_beg,args.train_part_end,args.tr_len,\
                                        args.img_size,args.resize_image,args.exp_id)
    sampler = Sampler(len(train_dataset.videoPaths),train_dataset.nbImages)
    trainLoader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=args.batch_size,sampler=sampler, collate_fn=collateSeq, # use custom collate function here
                      pin_memory=False,num_workers=args.num_workers)

    return trainLoader,train_dataset

def findVideos(propStart,propEnd):

    videoPaths = sorted(glob.glob("../data/*.avi"))
    videoPaths = np.array(videoPaths)[int(propStart*len(videoPaths)):int(propEnd*len(videoPaths))]

    return videoPaths

def getGT(vidName):
    ''' For one video, returns the label of each frame

    Args:
    - vidName (str): the video name. It is the name of the video file minus the extension.
    Returns:
    - gt (array): the list of labels corresponding to each image

    '''

    if not os.path.exists("../data/annotations/{}_targ.csv".format(vidName)):

        phases = np.genfromtxt("../data/annotations/{}_phases.csv".format(vidName),dtype=str,delimiter=",")

        gt = np.zeros((int(phases[-1,-1])+1))

        for phase in phases:
            #The dictionary called here convert the label into a integer

            gt[int(phase[1]):int(phase[2])+1] = formatData.getLabels()[phase[0]]

        np.savetxt("../data/annotations/{}_targ.csv".format(vidName),gt)
    else:
        gt = np.genfromtxt("../data/annotations/{}_targ.csv".format(vidName))

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

    argreader.parser.add_argument('--img_size', type=int,metavar='WIDTH',
                        help='The size of each edge of the resized images, if resize_image is True, else, the size of the image')

    argreader.parser.add_argument('--train_part_beg', type=float,metavar='START',
                        help='The (normalized) start position of the dataset to use for training')
    argreader.parser.add_argument('--train_part_end', type=float,metavar='END',
                        help='The (normalized) end position of the dataset to use for training')
    argreader.parser.add_argument('--val_part_beg', type=float,metavar='START',
                        help='The (normalized) start position of the dataset to use for validation')
    argreader.parser.add_argument('--val_part_end', type=float,metavar='END',
                        help='The (normalized) end position of the dataset to use for validation')
    argreader.parser.add_argument('--test_part_beg', type=float,metavar='START',
                        help='The (normalized) start position of the dataset to use for testing')
    argreader.parser.add_argument('--test_part_end', type=float,metavar='END',
                        help='The (normalized) end position of the dataset to use for testing')

    argreader.parser.add_argument('--resize_image', type=args.str2bool, metavar='S',
                        help='to resize the image to the size indicated by the img_width and img_heigth arguments.')

    argreader.parser.add_argument('--class_nb', type=int, metavar='S',
                        help='The number of class of to model')

    return argreader

if __name__ == "__main__":

    train_part_beg = 0
    train_part_end = 0.5
    val_part_beg = 0.5
    val_part_end = 1

    tr_len = 5
    val_l = 5
    img_size = 500
    resize_image = False
    exp_id = "Test"

    batch_size = 5
    num_workers = 1

    '''
    train_dataset = SeqTrDataset(train_part_beg,train_part_end,tr_len,\
                                        img_size,resize_image,exp_id)
    sampler = Sampler(len(train_dataset.videoPaths))
    trainLoader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,sampler=sampler, collate_fn=collateSeq, # use custom collate function here
                      pin_memory=False,num_workers=num_workers)

    for batch in trainLoader:
        print(batch[0].shape,batch[1].shape,batch[2])
        sys.exit(0)
    '''

    '''
    valLoader = TestLoader(val_l,val_part_beg,val_part_end,\
                                        img_size,resize_image,\
                                        exp_id)

    for batch in valLoader:
        print(batch[0].shape,batch[1].shape,batch[2],batch[3])
        sys.exit(0)

    '''
