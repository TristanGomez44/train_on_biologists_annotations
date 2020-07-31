
from args import ArgReader

import random
import torchvision
import glob
import os
import numpy as np
import cv2
import warnings
from shutil import copyfile
from scipy.io import loadmat
import subprocess
import sys
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

    makeDataset(trainPaths,"train",prop=1.0/5)
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

			if os.path.exists("../data/big/annotations/{}_phases.csv".format(vidName)):
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

def formatAircraft(path):

	setDict = {"trainval":"train","test":"test"}

	for set in ["trainval","test"]:
		print(set)
		rows = np.genfromtxt(os.path.join(path,"images_variant_{}.txt".format(set)),delimiter=",",dtype=str)

		for i in range(len(rows)):

			if i%600 == 0:
				print("\t",i)

			splitted_row = rows[i].split(" ")
			name = splitted_row[0]
			label = " ".join(splitted_row[1:])

			if not os.path.exists("../data/aircraft_{}/{}/".format(setDict[set],label)):
				os.makedirs("../data/aircraft_{}/{}/".format(setDict[set],label))

			copyfile(os.path.join(path,"images","{}.jpg".format(name)), "../data/aircraft_{}/{}/{}.jpg".format(setDict[set],label,name))

def formatCars(imgsPath,annotPath):

	if not os.path.exists("../data/cars_train"):
		subprocess.call("tar -xzf {} -C ../data/".format(imgsPath),shell=True)

	if not os.path.exists("../data/cars_train"):
		os.makedirs("../data/cars_train")
	if not os.path.exists("../data/cars_test"):
		os.makedirs("../data/cars_test")

	mat = loadmat(annotPath)
	for i,row in enumerate(mat["annotations"][0]):

		if i%100 == 0:
			print(i)

		path,label,isTrain = row[0].item(),row[-2].item(),row[-1].item()
		folder = "cars_train" if isTrain else "cars_test"
		filename = os.path.basename(path)

		if not os.path.exists("../data/{}/{}/{}".format(folder,label,filename)):

			if not os.path.exists("../data/{}/{}/".format(folder,label)):
				os.makedirs("../data/{}/{}/".format(folder,label))

			copyfile("../data/"+path,"../data/{}/{}/{}".format(folder,label,filename))

def formatDogs(imgsPath,trainTestSplitPath):

	if not os.path.exists("../data/Images/"):
		subprocess.call("tar -xf {} -C ../data/".format(imgsPath),shell=True)
	if not os.path.exists("../data/train_list.mat"):
		subprocess.call("tar -xf {} -C ../data/".format(trainTestSplitPath),shell=True)

	trainImgs = [ row[0].item() for row in loadmat("../data/train_list.mat")["annotation_list"] ]
	testImgs = [ row[0].item() for row in loadmat("../data/test_list.mat")["annotation_list"] ]

	imgsList = [trainImgs,testImgs]
	folders = ["dogs_train","dogs_test"]

	for i,folder in enumerate(folders):

		imgs = imgsList[i]

		for path in imgs:

			label,filename = path.split("/")

			if not os.path.exists("../data/{}/{}/".format(folder,label)):
				os.makedirs("../data/{}/{}/".format(folder,label))

			copyfile("../data/Images/"+path+".jpg","../data/{}/{}/{}.jpg".format(folder,label,filename))


def main(argv=None):

	#Getting arguments from config file and command row
	#Building the arg reader
	argreader = ArgReader(argv)

	argreader.parser.add_argument('--embryo',action="store_true",help='To format the embryo dataset')
	argreader.parser.add_argument('--aircraft',type=str,help='To format the aircraft dataset',metavar="PATH",default="")
	argreader.parser.add_argument('--cars',type=str,nargs=2,help='To format the cars dataset',metavar="PATH")
	argreader.parser.add_argument('--dogs',type=str,nargs=2,help='To format the dogs dataset',metavar="PATH")

	#Reading the comand row arg
	argreader.getRemainingArgs()

	#Getting the args from command row and config file
	args = argreader.args

	if args.embryo:
		formatEmbryo()
	if args.aircraft != "":
		formatAircraft(args.aircraft)
	if not args.cars is None:
		formatCars(*args.cars)
	if not args.dogs is None:
		formatDogs(*args.dogs)
if __name__ == "__main__":
    main()
