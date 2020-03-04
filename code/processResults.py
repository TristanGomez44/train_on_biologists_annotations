
from args import ArgReader
from args import str2bool
import os
import glob

import torch
import numpy as np

from skimage.transform import resize
import matplotlib.pyplot as plt
plt.switch_backend('agg')

from sklearn.manifold import TSNE
import sklearn
import matplotlib.cm as cm
import matplotlib.patches as patches
import pims
import cv2
from PIL import Image

import load_data


import metrics
import utils
import formatData
import trainVal
import formatData
import scipy
import sys

import configparser

import matplotlib.patheffects as path_effects
import imageio
from skimage import img_as_ubyte

from scipy import stats
import math
from PIL import Image
from PIL import Image, ImageEnhance

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

def plotPointsImageDataset(imgNb,redFact,plotDepth,args):

    cm = plt.get_cmap('plasma')

    exp_id = args.exp_id
    model_id = args.model_id

    pointPaths = sorted(glob.glob("../results/{}/points_{}_epoch*_.npy".format(exp_id,model_id)),key=utils.findNumbers)

    points = np.concatenate(list(map(lambda x:np.load(x)[:imgNb][:,:,:][np.newaxis],pointPaths)),axis=0)
    points = np.transpose(points, axes=[1,0,2,3])

    imgLoader = load_data.buildSeqTestLoader(args,"val",normalize=False)

    batchNb = imgNb//args.val_batch_size
    totalImgNb = 0

    for batchInd,(imgBatch,_) in enumerate(imgLoader):
        print(batchInd,imgBatch.size())
        for imgInd in range(len(imgBatch)):

            if totalImgNb<imgNb:
                print("\t",imgInd,"/",totalImgNb)
                print("\t","Writing video",imgInd)
                with imageio.get_writer("../vis/{}/points_{}_img{}_depth={}.mp4".format(exp_id,model_id,totalImgNb,plotDepth), mode='I',fps=20,quality=9) as writer:

                    img = imgBatch[imgInd].detach().permute(1,2,0).numpy().astype(float)

                    for epoch in range(len(points[imgInd])):

                        pts = points[imgInd,epoch]
                        mask = np.ones((img.shape[0]//redFact,img.shape[1]//redFact,3)).astype("float")

                        if plotDepth:
                            ptsValues = pts[:,2]
                        else:
                            ptsValues = np.abs(pts[:,3:]).sum(axis=-1)

                        ptsValues = cm(ptsValues/ptsValues.max())[:,:3]
                        mask[pts[:,1].astype(int),pts[:,0].astype(int)] = ptsValues

                        mask = resize(mask, (img.shape[0],img.shape[1]),anti_aliasing=True,mode="constant",order=0)

                        imgMasked = img*255*mask

                        imgMasked = Image.fromarray(imgMasked.astype("uint8"))
                        draw = ImageDraw.Draw(imgMasked)
                        draw.text((0, 0),str(epoch),(0,0,0))
                        imgMasked = np.asarray(imgMasked)
                        cv2.imwrite("testProcessResults.png",imgMasked)
                        writer.append_data(img_as_ubyte(imgMasked.astype("uint8")))

                    totalImgNb += 1
        if batchInd>=batchNb:
            break

def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    ####################### Plot points computed for a point net model #############################

    argreader.parser.add_argument('--plot_points_image_dataset',action="store_true",help='To plot the points computed by the visual module of a point net model \
                                    on an image dataset. The -c (config file), --image_nb and --reduction_fact arg must be set.')

    argreader.parser.add_argument('--image_nb',type=int,metavar="INT",help='For the --plot_points_image_dataset arg. \
                                    The number of images to plot the points of.')

    argreader.parser.add_argument('--reduction_fact',type=int,metavar="INT",help='For the --plot_points_image_dataset arg.\
                                    The reduction factor of the point cloud size compared to the full image. For example if the image has size \
                                    224x224 and the points cloud lies in a 56x56 frame, this arg should be 224/56=4')

    argreader.parser.add_argument('--plot_depth',type=str2bool,metavar="BOOL",help='For the --plot_points_image_dataset arg. Plots the depth instead of point feature norm.')

    argreader = load_data.addArgs(argreader)

    #Reading the comand line arg
    argreader.getRemainingArgs()

    #Getting the args from command line and config file
    args = argreader.args

    if args.plot_points_image_dataset:
        plotPointsImageDataset(args.image_nb,args.reduction_fact,args.plot_depth,args)

if __name__ == "__main__":
    main()
