
from args import ArgReader

import random
import torchvision
import glob
import os
import numpy as np
import cv2
import warnings

labelDict = {"tPB2":0,"tPNa":1,"tPNf":2,"t2":3,"t3":4,"t4":5,"t5":6,"t6":7,"t7":8,"t8":9,"t9+":10,"tM":11,"tSB":12,"tB":13,"tEB":14,"tHB":15}

def getNoAnnotVideos():
	return np.genfromtxt("../data/noAnnot.csv",dtype=str,delimiter=",")

def getTooFewPhaseVideos(nbMinimumPhases):

	tooFewPhaseVids = []

	for vidPath in sorted(glob.glob("../data/*/*.avi")):

		vidName = os.path.splitext(os.path.basename(vidPath))[0]
		annotPath = os.path.join(os.path.dirname(vidPath),"annotations","{}_phases.csv".format(vidName))

		if os.path.exists(annotPath) and os.stat(annotPath).st_size > 0:
			csv = np.genfromtxt(annotPath,delimiter=",")

			if len(csv.shape) < 2 or len(csv) < nbMinimumPhases:
				tooFewPhaseVids.append(vidName)

	return tooFewPhaseVids

def getEmptyAnnotVideos():

	videoPaths = sorted(glob.glob("../data/*/*.avi"))
	emptyAnnotVids = []

	with warnings.catch_warnings():
		warnings.filterwarnings('error')
		for vidPath in videoPaths:
			vidName = os.path.basename(os.path.splitext(vidPath)[0])
			phasesPath = os.path.join(os.path.dirname(vidPath),"annotations",vidName+"_phases.csv")
			if os.path.exists(phasesPath):
				if os.stat(phasesPath).st_size == 0:
					emptyAnnotVids.append(vidName)

	return emptyAnnotVids

def removeVid(videoPaths,videoToRemoveNames):
    #Removing videos with bad format
    vidsToRemove = []
    for vidPath in videoPaths:
        for vidName in videoToRemoveNames:
            if os.path.splitext(os.path.basename(vidPath))[0] == vidName:
                vidsToRemove.append(vidPath)
    for vidPath in vidsToRemove:
        videoPaths.remove(vidPath)

    return videoPaths

def formatEmbryo():

    allPaths = sorted(glob.glob("../data/big/*avi"))

    allPaths = removeVid(allPaths,getNoAnnotVideos())
    allPaths = removeVid(allPaths,getEmptyAnnotVideos())
    allPaths = removeVid(allPaths,getTooFewPhaseVideos(6))

    random.seed(0)
    random.shuffle(allPaths)

    trainPaths = allPaths[:len(allPaths)//2]
    testPaths = allPaths[len(allPaths)//2:]

    #makeDataset(trainPaths,"train",prop=1.0/5)
    makeDataset(testPaths,"test",prop=1.0/5)

def makeDataset(paths,mode,prop):

    if not os.path.exists("../data/embryo_img_{}/".format(mode)):
        os.makedirs("../data/embryo_img_{}/".format(mode))

    for label in labelDict.keys():
        if not os.path.exists("../data/embryo_img_{}/{}/".format(mode,label)):
            os.makedirs("../data/embryo_img_{}/{}/".format(mode,label))

    print(mode)
    for k,path in enumerate(paths):

        timeStamps = torchvision.io.read_video_timestamps(path,pts_unit="sec")[0]
        vidName = os.path.splitext(os.path.basename(path))[0]
        print("\t",vidName,k,"/",len(paths))

        if len(glob.glob("../data/embryo_img_{}/*/{}*".format(mode,vidName))) < len(timeStamps)*prop:

            annot = np.genfromtxt("../data/big/annotations/{}_phases.csv".format(vidName),delimiter=",",dtype="str")

            startFr = int(annot[0,1])

            endFr = -1
            while endFr < len(timeStamps) - 1:

                endFr = startFr + 600
                if endFr > len(timeStamps):
                    endFr = len(timeStamps) - 1

                startTime = timeStamps[startFr]
                endTime = timeStamps[endFr]

                frames = torchvision.io.read_video(path,pts_unit="sec",start_pts=startTime,end_pts=endTime)[0]
                #Removing top border
                if frames.size(1) > frames.size(2):
                    frames = frames[:,frames.size(1)-frames.size(2):]

                #Removing time
                frames[:,-30:] = 0

                for i,frame in enumerate(frames):
                    if i%(1/prop) == 0:
                        label = getClass(annot,startFr+i)
                        cv2.imwrite("../data/embryo_img_{}/{}/{}_{}.png".format(mode,label,vidName,startFr+i),frame.numpy())

                startFr = endFr + 1

def getClass(annot,i):
    for row in annot:

        if int(row[1])<=i and i<=int(row[2]):
            return row[0]

    raise ValueError("Couldn't find label for annot file",annot,",frame ",i)

def main(argv=None):

    #Getting arguments from config file and command row
    #Building the arg reader
    argreader = ArgReader(argv)

    #Reading the comand row arg
    argreader.getRemainingArgs()

    #Getting the args from command row and config file
    args = argreader.args

    formatEmbryo()

if __name__ == "__main__":
    main()
