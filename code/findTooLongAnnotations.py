import os
import glob
import numpy
import utils
import numpy as np

vidPaths = sorted(glob.glob("../data/big/*avi"))

for i,vidPath in enumerate(vidPaths):

    if i%1 == 0:
        print(i,"/",len(vidPaths),":",vidPath)

    videoName = os.path.splitext(os.path.basename(vidPath))[0]

    if os.path.exists("../data/big/annotations/{}_phases.csv".format(videoName)):
        frameNb = utils.getVideoFrameNb(vidPath)
        lastFrameAnnot =np.genfromtxt("../data/big/annotations/{}_phases.csv".format(videoName),delimiter=",")[-1,-1]
        if frameNb != lastFrameAnnot:
            print(videoName,frameNb,lastFrameAnnot)
