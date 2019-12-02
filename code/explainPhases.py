import cv2
import imageio
import os
import formatData
import load_data
import numpy as np
from skimage import transform,io
from skimage import img_as_ubyte
import utils
import sys


def writeText(text,frameWithCaption,bottomLeftCornerOfText,font,fontScale,fontColor,lineType):

    frameWithCaption = cv2.putText(frameWithCaption,text,
                                   bottomLeftCornerOfText,
                                   font,
                                   fontScale,
                                   fontColor,
                                   lineType)

    return frameWithCaption


font                   = cv2.FONT_HERSHEY_SIMPLEX
fontScale              = 1
fontColorModel              = (255,255,255)
fontColorGT              = (0,0,0)
lineType               = 2

imgSize = 216,384

dataset = "small"
videoName = "ALR493_EMB6_TRANSFER"
videoPath = "../data/{}/{}.avi".format(dataset,videoName)

cap = cv2.VideoCapture(videoPath)
ret, frame = cap.read()

frameInd = 0

gt = load_data.getGT(videoName,dataset)
revLabelDict = formatData.getReversedLabels()

print("Nb frames : ",utils.getVideoFrameNb(videoPath))
print("Frame rate", utils.getVideoFPS(videoPath))
print("Target size : ",len(gt))

with imageio.get_writer('../vids/phases.mp4',fps=30,quality=5) as writer:
    while ret:

        frameWithCaption = np.ones((frame.shape[0]+90,frame.shape[1],3))*255
        frameWithCaption[:frame.shape[0]] = frame.astype("uint8")

        bottomLeftCornerOfText = (0,frameWithCaption.shape[1]+45)

        if frameInd < len(gt):

            writeText('Developpement phase : {}'.format(revLabelDict[gt[frameInd]]),frameWithCaption,bottomLeftCornerOfText,font,fontScale,fontColorGT,lineType)

            writer.append_data(img_as_ubyte(frameWithCaption.astype("uint8")))

            ret, frame = cap.read()
            frameInd += 1
        else:
            ret = False
