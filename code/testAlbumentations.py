
import albumentations
from albumentations import Compose
import pims
import numpy as np
import cv2

def removeTopFunc(x):
    #Removing the top part where the name of the video is written
    x = x[x.shape[0]-500:]
    return x

vid = pims.Video("../data/big/CV306-2.avi")

transf = Compose([
        albumentations.RandomGridShuffle(grid=(5, 5), p=1.0),
    ], p=1)

for i in range(20):

    frame = vid[i]
    frame = removeTopFunc(frame)
    frame = transf(image=frame)["image"]

    print(frame.shape)
    cv2.imwrite("../data/frame_{}.png".format(i),frame)
