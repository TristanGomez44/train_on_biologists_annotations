import cv2
import numpy as np
import imageio
import pims
import os
from skimage import img_as_ubyte
import utils
import load_data
import glob

from skimage import data, color
from skimage.transform import hough_circle, hough_circle_peaks
from skimage.feature import canny
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt

from scipy.stats import trim_mean

def create_circular_mask(h, w, center, radius):
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask

videoPaths = sorted(glob.glob("../data/big/*avi"))
videoInd = 0
nbCircles = 1

while videoInd < 4:

    videoPath = videoPaths[videoInd]
    print(videoPath)
    videoName = os.path.basename(videoPath)
    resultVideoPath = os.path.join("../data/",videoName.replace(".avi","_circDet.avi"))
    video = pims.Video(videoPath)
    distList,varCoeffList = [],[]
    runCentX,runCentY,runRadi = None,None,None

    with imageio.get_writer(resultVideoPath, mode='I',fps=25) as writer:
        frameInd=0
        frameNb = utils.getVideoFrameNb(videoPath)

        while frameInd < frameNb:
            frame = video[frameInd]
            frame = frame[frame.shape[0]-500:]
            bigFrame = np.zeros((512,512,3)).astype("uint8")
            bigFrame[:frame.shape[0],:frame.shape[1]] = frame

            circles = cv2.HoughCircles(bigFrame[...,0],cv2.HOUGH_GRADIENT,1,200,param1=50,param2=30,minRadius=100,maxRadius=120)

            if not circles is None:
                if runCentX is None:
                    runCentX = circles[0,0][0]
                    runCentY = circles[0,0][1]
                    runRadi = circles[0,0][2]
                else:
                    distList.append(np.abs(circles[0,0][0]-runCentX)+np.abs(circles[0,0][1]-runCentY))

                    if np.abs(circles[0,0][0]-runCentX)+np.abs(circles[0,0][1]-runCentY) < 60:
                        runCentX = 0.9*runCentX+0.1*circles[0,0][0]
                        runCentY = 0.9*runCentY+0.1*circles[0,0][1]
                        runRadi = 0.9*runRadi+0.1*circles[0,0][2]
                    else:
                        runCentX = circles[0,0][0]
                        runCentY = circles[0,0][1]
                        runRadi = circles[0,0][2]

                cv2.circle(bigFrame,(int(runCentX),int(runCentY)),int(runRadi),(0,255,0),2)

                mask = create_circular_mask(bigFrame.shape[0], bigFrame.shape[1], (runCentX,runCentY), runRadi)
                maskedBigFrame = bigFrame*mask[:,:,np.newaxis]
                varCoeffList.append(maskedBigFrame.std()/trim_mean(maskedBigFrame.reshape(-1), 0.1, axis=0))

            writer.append_data(bigFrame)
            frameInd+=1

    videoInd += 1
    video = None

    plt.figure()
    plt.plot(distList)
    plt.savefig("../vis/{}_embryoCenterDist.png".format(videoName))

    plt.figure(figsize=(16,6))
    #ploting bar lines
    gt = load_data.getGT(videoName.replace(".avi",""),"big")
    gt = gt[1:]-gt[:-1]
    gt = gt.nonzero()[0]
    plt.bar(gt,np.array(varCoeffList).max() , width=0.8,bottom=np.array(varCoeffList).min(),color="red")
    #ploting text labels
    gt = np.genfromtxt("../data/big/annotations/{}_phases.csv".format(videoName.replace(".avi","")),dtype=str,delimiter=",")[1:,:2]
    for phase in gt:
        plt.text(int(phase[1]), 0, phase[0])
    plt.plot(varCoeffList)
    plt.tight_layout()
    plt.savefig("../vis/{}_grayVarCoeff.png".format(videoName))
