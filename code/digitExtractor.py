import numpy as np
import cv2
import glob
import os
import argparse
from sklearn.cluster import KMeans
import numpy as np
import sys


imgHeigth,imgWidth = 500,500

heigth,width = 15,8

#This dictionary indicates the position of all the digits visible in the videos :
# - The digits indicating time (except the hundreds digit because it can only be one or not exist)
# - The digits indicating the well index
pos = {"digit1"  :      {"Y1":475,"Y2":475+heigth,"X1":472,"X2":472+width}, \
	   "digit2" :       {"Y1":475,"Y2":475+heigth,"X1":462,"X2":462+width}, \
	   "digit3" :       {"Y1":475,"Y2":475+heigth,"X1":456,"X2":456+width}, \
	   "wellInd_dig1" : {"Y1":475,"Y2":475+heigth,"X1":37, "X2":37+width},  \
	   "wellInd_dig2" : {"Y1":475,"Y2":475+heigth,"X1":44, "X2":44+width}}
hundredsDigitPos =      {"Y1":475,"Y2":475+heigth,"X1":449,"X2":449+width}

misplacedOrNoDigitNorm = 70000
centeredDigitNorm = 35500

videosToRemove = ["BE150-1.avi","MZ900-8.avi"]

def getPos():
	return pos

def getVideosToRemove():
	return videosToRemove

def getDigitYPos():
	return pos["digit1"]["Y1"],pos["digit1"]["Y2"]

def writeWithCondition(condition,dataset,key,totalCountDict,countDict,digit):
	if condition:
		res = cv2.imwrite("../data/{}/timeImg/{}/{}.png".format(dataset,key,totalCountDict[key]),digit)
		totalCountDict[key] += 1
		countDict[key] += 1

def computeCondAndWrite(digit,goodDigitCond,goodDigit,cond,countDict,totalCountDict,key,imgNb,dataset):
	cond[key] = totalCountDict[key] < imgNb and goodDigitCond
	writeWithCondition(cond[key],dataset,key,totalCountDict,countDict,digit)
	if goodDigitCond:
		goodDigit = True

	return goodDigit

def initBigDatasetVariables(pos):

	origKeys = list(pos.keys())

	for key in origKeys:
		if key.find("digit") != -1:
			if key.find("digit1") != -1:
				pos[key+"_alt"] = {"Y1":pos[key]["Y1"]+6,"Y2":pos[key]["Y2"]+6, "X1":pos[key]["X1"]+2,"X2":pos[key]["X2"]+2}
			else:
				pos[key+"_alt"] = {"Y1":pos[key]["Y1"]+6,"Y2":pos[key]["Y2"]+6,"X1":pos[key]["X1"]+5,"X2":pos[key]["X2"]+5}
		#else:
		#	pos[key+"_alt"] = {"Y1":pos[key]["Y1"]+4,"Y2":pos[key]["Y2"]+4,"X1":pos[key]["X1"]+1,"X2":pos[key]["X2"]+1}

	pos["digit1_alt2"] = {"Y1":pos["digit1_alt"]["Y1"],"Y2":pos["digit1_alt"]["Y2"],"X1":pos["digit1_alt"]["X1"]+7,"X2":pos["digit1_alt"]["X2"]+7}
	pos["digit1_alt3"] = {"Y1":pos["digit1"]["Y1"]+5,  "Y2":pos["digit1"]["Y2"]+5,  "X1":pos["digit1"]["X1"]-3,    "X2":pos["digit1"]["X2"]-3}
	pos["digit2_alt2"] = {"Y1":pos["digit2_alt"]["Y1"],"Y2":pos["digit2_alt"]["Y2"],"X1":pos["digit2_alt"]["X1"]+5,"X2":pos["digit2_alt"]["X1"]+10}
	pos["digit2_alt3"] = {"Y1":pos["digit2_alt"]["Y1"],"Y2":pos["digit2_alt"]["Y2"],"X1":pos["digit2_alt"]["X1"]-8,"X2":pos["digit2_alt"]["X2"]-8}
	pos["digit3_alt"] = {"Y1":pos["digit3_alt"]["Y1"],"Y2":pos["digit3_alt"]["Y2"],"X1":pos["digit3_alt"]["X1"]-2,"X2":pos["digit3_alt"]["X2"]-3}

	pos["wellInd_dig2_alt"] = {"Y1":pos["wellInd_dig1"]["Y1"]+4,"Y2":pos["wellInd_dig1"]["Y2"]+4,"X1":pos["wellInd_dig1"]["X1"]+1,"X2":pos["wellInd_dig1"]["X2"]+1}
	pos["wellInd_dig1_alt"] = {"Y1":pos["wellInd_dig2_alt"]["Y1"],"Y2":pos["wellInd_dig2_alt"]["Y2"],"X1":pos["wellInd_dig2_alt"]["X1"]-7,"X2":pos["wellInd_dig2_alt"]["X2"]-7}

	#This is only useful for the dataset "big"
	goodDigitCondFuncDict = {"digit1":     lambda digit:digit.sum() > misplacedOrNoDigitNorm,
							 "digit1_alt": lambda digit:digit[-1,:,:].sum() < 3000 and digit[:,0,:].sum() > 9000  and digit[:,-1,:].sum() > 9500,
							 "digit1_alt2":lambda digit:digit[-1,:,:].sum() < 3000 and digit[-3,:,:].sum() > 3000 and digit[:,-1,:].sum() > 8000,
							 "digit2":     lambda digit:digit.sum() > misplacedOrNoDigitNorm,
							 "digit2_alt" :lambda digit:digit[-1,:,:].sum() < 3000 and digit[-3,:,:].sum() > 3000 and digit[:,-1,:].sum() > 9000 and digit[:9,:2,:].sum() < 13000,
							 "digit2_alt2":lambda digit:digit[:,-2:,:].sum() < 18000,
							 "digit2_alt3":lambda digit:True,
							 "digit3":     lambda digit:digit.sum() > misplacedOrNoDigitNorm,
							 "digit3_alt":lambda digit:digit[-1,:,:].sum() < 3000 and digit[-3,:,:].sum() > 3000}

	return pos,goodDigitCondFuncDict

def clusterDigits(dataset,imgNb):
	"""	Extract digits from embryo images in the ../data/ folder and cluster them using k-means

	Once it is finished running, in ../data/ a folder "timeImg" is created and contains several folders : digit1, digit2, digit3, digit4, wellInd, etc.
	Each folder is dedicated to one digit visible in the image. It can be the well index (bottom left of the image), or one of the four digits
	indicating the hour at which the image was taken (bottom right). In each of those 4 folder you will find 10 folders, each containing a cluster
	of digits. You have to rename manually those folders with the correct class. E.g., the folder containing all the "1" should be renamed "1".

	Args:
		- imgNb (int) : the number of digit crops to extract. For the digits indicating the time, they will be extracted from all the frame of the first videos and for the digit
						indicating the well id, only the first few frames of all the videos will be extracted.

	"""

	if dataset == "big":
		#Some video in the dataset "big" have a slightly different digit position

		pos,goodDigitCondFuncDict = initBigDatasetVariables(getPos())

		clustNb = {}
		for key in pos.keys():
			if key.find("wellInd_dig2") != -1:
				clustNb[key] = 3
			else:
				clustNb[key] = 10

		clustNb = {"digit1":10,"digit1_alt":10,"digit1_alt2":10,"digit1_alt3":10,
				   "digit2":10,"digit2_alt":10,"digit2_alt2":10,"digit2_alt3":10,
				   "digit3":10,"digit3_alt":10,
				   "wellInd_dig1":9,"wellInd_dig1_alt":2,
				   "wellInd_dig2":3,"wellInd_dig2_alt":9}

		#This dict map cluster index to the real class (i.e. the digit)
		clustInd2Digit = {"digit1":{0:5,1:1,2:3,3:7,4:9,5:2,6:4,7:8,8:0,9:6},
						  "digit1_alt":{0:3,1:4,2:0,3:2,4:7,5:6,6:5,7:1,8:9,9:8},
						  "digit1_alt2":{0:9,1:1,2:3,3:5,4:7,5:4,6:2,7:0,8:6,9:8},
						  "digit1_alt3":{0:4,1:6,2:2,3:0,4:1,5:7,6:8,7:9,8:5,9:3},
						  "digit2":{0:3,1:1,2:9,3:5,4:7,5:2,6:4,7:8,8:0,9:6},
				  		  "digit2_alt":{0:6,1:1,2:2,3:7,4:4,5:9,6:0,7:3,8:5,9:8},
				  		  "digit2_alt2":{0:7,1:3,2:1,3:4,4:0,5:2,6:6,7:9,8:5,9:8},
				  		  "digit2_alt3":{0:6,1:1,2:2,3:7,4:4,5:9,6:3,7:5,8:8,9:0},
						  "digit3":{0:0,1:1,2:2,3:3,4:7,5:4,6:5,7:8,8:6,9:9},
						  "digit3_alt":{0:1,1:8,2:7,3:2,4:9,5:4,6:5,7:3,8:6,9:0},
						  "wellInd_dig1":{0:2,1:1,2:6,3:3,4:4,5:7,6:5,7:9,8:8},
						  "wellInd_dig2_alt":{0:6,1:4,2:2,3:7,4:1,5:3,6:9,7:0,8:5},
						  "wellInd_dig2":{0:2,1:1,2:0},
						  "wellInd_dig1_alt":{0:0,1:1}}

	else:

		clustNb = {"digit1":10,"digit2":10,"digit3":10,"wellInd_dig1":9,"wellInd_dig2":3}

		#This dict map cluster index to the real class (i.e. the digit)
		clustInd2Digit = {"digit1":{0:9, 1:1, 2:7, 3:6, 4:8, 5:2, 6:4, 7:0, 8:5, 9:3},
						  "digit2":{0:9, 1:4, 2:3, 3:5, 4:7, 5:1, 6:2, 7:8, 8:0, 9:6},
						  "digit3":{0:4, 1:3, 2:5, 3:1, 4:7, 5:2, 6:0, 7:8, 8:9, 9:6},
						  "wellInd_dig1":{0:8, 1:1, 2:2, 3:5, 4:7, 5:4, 6:9, 7:6, 8:3},
						  "wellInd_dig2":{0:0, 1:1, 2:2}}

	if not os.path.exists("../data/{}/timeImg/".format(dataset)):
		print("Extracting digit from the images...")

		#if not os.path.exists("../data/{}/timeImg/".format(dataset)):
		vidPaths = glob.glob("../data/{}/*avi".format(dataset))

		#Removing the videos with big problems
		for vidName in videosToRemove:
			vidPaths.remove("../data/{}/".format(dataset)+vidName)

		vidInd = 0

		#Store the total number of cropped images written for each digit
		totalCountDict = {}
		for key in pos.keys():
			totalCountDict[key] = 0

		#Store the number of images written for each digit in the current video
		countDict = {}

		if not os.path.exists("../data/{}/timeImg/".format(dataset)):
			os.makedirs("../data/{}/timeImg/".format(dataset))

		for key in pos.keys():
			if not os.path.exists("../data/{}/timeImg/{}/".format(dataset,key)):
				os.makedirs("../data/{}/timeImg/{}/".format(dataset,key))

		wellIndCropPerVid = 1 + imgNb//len(vidPaths)

		while vidInd < len(vidPaths):
			#The number of images decoded in the current video
			videoImgCount = 0
			print(vidInd,"/",len(vidPaths)," : ",vidPaths[vidInd])
			cap = cv2.VideoCapture(vidPaths[vidInd])

			for key in pos.keys():
				countDict[key] = 0

			ret, frame = cap.read()
			frameInd = 0
			enoughImgParsed = False
			while ret and not enoughImgParsed:
				frame = frame[frame.shape[0]-imgHeigth:,:]
				frameInd += 1

				goodDigit1,goodDigit2,goodDigit3 = False,False,False
				cond = {}

				for key in pos.keys():

					digit = frame[pos[key]["Y1"]:pos[key]["Y2"],pos[key]["X1"]:pos[key]["X2"],:]

					if dataset == "big":

						if key == "digit1" or ((key == "digit1_alt" or key == "digit1_alt2") and (not goodDigit1)):
							goodDigitCond = goodDigitCondFuncDict[key](digit)
							goodDigit1 = computeCondAndWrite(digit,goodDigitCond,goodDigit1,cond,countDict,totalCountDict,key,imgNb,dataset)
						if key == "digit1_alt3" and (not goodDigit1):
							goodDigit1 = computeCondAndWrite(digit,True,goodDigit1,cond,countDict,totalCountDict,key,imgNb,dataset)

						if key =="digit2" or (key == "digit2_alt" and (not goodDigit2)):
							goodDigitCond = goodDigitCondFuncDict[key](digit)
							goodDigit2 = computeCondAndWrite(digit,goodDigitCond,goodDigit2,cond,countDict,totalCountDict,key,imgNb,dataset)
						if (key == "digit2_alt2" or key == "digit2_alt3") and (not goodDigit2):
							goodDigitCond = goodDigitCondFuncDict[key](digit)
							goodDigit2 = computeCondAndWrite(digit,goodDigitCond,goodDigit2,cond,countDict,totalCountDict,key,imgNb,dataset)
						if key == "digit3" or (key == "digit3_alt" and (not goodDigit3)):
							goodDigitCond = goodDigitCondFuncDict[key](digit)
							goodDigit3 = computeCondAndWrite(digit,goodDigitCond,goodDigit3,cond,countDict,totalCountDict,key,imgNb,dataset)

						if key.find("wellInd_dig") != -1:
							goodDigit = digit.sum() > misplacedOrNoDigitNorm and countDict[key] < wellIndCropPerVid
							cond[key] = totalCountDict[key] < imgNb and goodDigit
							writeWithCondition(cond[key],dataset,key,totalCountDict,countDict,digit)

					else:

						if (key == "digit3" or key == "wellInd_dig2"):
							goodDigit = digit.sum() > blankDigitNorm
						if key != "digit3" and key != "wellInd_dig2":
							goodDigit = True

						cond[key] = totalCountDict[key] < imgNb and goodDigit
						writeWithCondition(cond[key],dataset,key,totalCountDict,countDict,digit)

				if dataset == "big":
					if (not goodDigit1):
						cv2.imwrite("../data/big/timeImg/vid{}_dig1.png".format(vidInd),frame)
						for key in pos.keys():
							if key.find("digit1") != -1:
								digit = frame[pos[key]["Y1"]:pos[key]["Y2"],pos[key]["X1"]:pos[key]["X2"],:]
								print(key,digit[-1,:,:].sum() < 3000,digit[-3,:,:].sum() > 3000)
								cv2.imwrite("../data/big/timeImg/vid{}_{}.png".format(vidInd,key),digit)
						sys.exit(0)

					if (not goodDigit2):
						cv2.imwrite("../data/big/timeImg/vid{}_dig2.png".format(vidInd),frame)
						for key in pos.keys():
							if key.find("digit2") != -1:
								digit = frame[pos[key]["Y1"]:pos[key]["Y2"],pos[key]["X1"]:pos[key]["X2"],:]
								cv2.imwrite("../data/big/timeImg/vid{}_{}.png".format(vidInd,key),digit)
						sys.exit(0)

					if (not goodDigit3) and frameInd>100:
						cv2.imwrite("../data/big/timeImg/vid{}_dig3.png".format(vidInd),frame)
						for key in pos.keys():
							if key.find("digit3") != -1:
								digit = frame[pos[key]["Y1"]:pos[key]["Y2"],pos[key]["X1"]:pos[key]["X2"],:]
								cv2.imwrite("../data/big/timeImg/vid{}_{}.png".format(vidInd,key),digit)
						sys.exit(0)

					#if sum([cond[key] for key in cond.keys() if key.find("digit") != -1]) == 0 and frameInd == 10:
					#	enoughImgParsed = True
					writtenDigit1 = sum([cond[key] for key in cond.keys() if key.find("digit1") != -1]) > 0
					writtenDigit2 = sum([cond[key] for key in cond.keys() if key.find("digit2") != -1]) > 0
					writtenDigit3 = sum([cond[key] for key in cond.keys() if key.find("digit3") != -1]) > 0

					if (not writtenDigit1) and (not writtenDigit2) and (not writtenDigit3) and frameInd > 10:
						enoughImgParsed = True

				ret, frame = cap.read()

			vidInd += 1

	print("Clustering the digits ... ")
	for key in pos.keys():
		imgPaths = glob.glob("../data/{}/timeImg/{}/*.png".format(dataset,key))
		data = []

		for path in imgPaths:
			img = cv2.imread(path)
			data.append(img.reshape((-1))[np.newaxis])

		if len(data) > 0:
			data = np.concatenate(data,axis=0)
			kmeans = KMeans(n_clusters=clustNb[key], random_state=0).fit(data)

			#Moving the image in their respective folder
			for i,imgPath in enumerate(imgPaths):
				if not os.path.exists("../data/{}/timeImg/{}/{}/".format(dataset,key,kmeans.labels_[i])):
					os.makedirs("../data/{}/timeImg/{}/{}/".format(dataset,key,kmeans.labels_[i]))
				fileName = os.path.basename(imgPath)
				os.rename(imgPath,"../data/{}/timeImg/{}/{}/{}".format(dataset,key,kmeans.labels_[i],fileName))

	#sys.exit(0)
	print("Renaming the folder according to the digits they contain")
	for key in pos.keys():

		#Renaming the folders and adding a "_" not there is no name conflict
		folders = sorted(glob.glob("../data/{}/timeImg/{}/*/".format(dataset,key)))
		for folder in folders:
			baseDir = "/".join(folder.split("/")[:-2])
			dirName = folder.split("/")[-2]
			clustInd = int(dirName)
			os.rename(folder,baseDir+"/"+str(clustInd2Digit[key][clustInd])+"_")

		#Removing the "_"
		folders = sorted(glob.glob("../data/{}/timeImg/{}/*/".format(dataset,key)))
		for folder in folders:
			baseDir = "/".join(folder.split("/")[:-2])
			dirName = folder.split("/")[-2]
			os.rename(folder,baseDir+"/"+dirName.replace("_",""))

	print("Done ! Don't forget to quickly check if the croped digits in timeImg correspond to the name of the folder they are in. Eg. : the digits in the folder '1' should all be 1s.")

class DigitIdentifier:
	""" This class extracts the digits of an image

	First the digits on the images are cropped and then they identified with the KNN algorithm.
	Each crop is compared to 10 digits of each class and is assigned to the group from which it is the closest

	Args:
	- neigbhorsNb (int): the number of neigbhors to compare with

	"""

	def __init__(self,dataset,neigbhorsNb=10):


		super(DigitIdentifier,self).__init__()

		self.refDict = {}
		self.dataset = dataset
		self.pos,self.goodDigitCondFuncDict = initBigDatasetVariables(pos)

		for digName in self.pos.keys():

			self.refDict[digName] = {}

			minSize = np.inf
			for label in sorted(glob.glob("../data/{}/timeImg/{}/*/".format(dataset,digName))):
				#This is a 2D matrix where each row is one example of the class
				imgList = np.concatenate(list(map(lambda x:cv2.imread(x).reshape((-1))[np.newaxis],sorted(glob.glob(label+"/*.png".format(digName)))[:neigbhorsNb])),axis=0)
				self.refDict[digName][label] = imgList

				#There are some labels for which there is fewer examples than other labels
				#It is necessary to select x examples for each labels, where x is the minimum number of examples for any label
				if len(imgList) < minSize:
					minSize = len(imgList)
			for label in sorted(glob.glob("../data/{}/timeImg/{}/*/".format(dataset,digName))):
				self.refDict[digName][label] = self.refDict[digName][label][:minSize]

		self.lastTime = None

	def findDigits(self,img,newVid=False):
		''' Crop the digits on an image and identify them with KNN algorithm

		The image contains two information  :
		- The well index, on the bottom left
		- The time at which the photo was taken, on the bottom right

		Args:
		- img (3D array): the image
		- newVid (bool): a boolean indicating if the image comes from a different video than the preceding frame
		Returns:
		- resDict (dict): a dictionary containing two keys :
			- "wellInd" : the index of the well
			- "time" : the time at which the image was taken

		'''

		#Cropping the upper part of the image
		img = img[img.shape[0]-imgHeigth:,:]

		rawResDict = {}
		cond = {}
		goodDigit1,goodDigit2,goodDigit3 = False,False,False

		for digName in self.pos.keys():
			digit = img[self.pos[digName]["Y1"]:self.pos[digName]["Y2"],self.pos[digName]["X1"]:self.pos[digName]["X2"],:]

			if self.dataset == "big":

				if digName == "digit1" or ((digName == "digit1_alt" or digName == "digit1_alt2") and (not goodDigit1)):
					goodDigitCond = self.goodDigitCondFuncDict[digName](digit)
				if digName == "digit1_alt3" and (not goodDigit1):
					goodDigitCond = True
				if digName.find("digit1") != -1 and goodDigitCond:
					goodDigit1 = True

				if digName =="digit2" or (digName == "digit2_alt" and (not goodDigit2)):
					goodDigitCond = self.goodDigitCondFuncDict[digName](digit)
				if (digName == "digit2_alt2" or digName == "digit2_alt3") and (not goodDigit2):
					goodDigitCond = self.goodDigitCondFuncDict[digName](digit)
				if digName.find("digit2") != -1 and goodDigitCond:
					goodDigit2 = True

				if digName == "digit3" or (digName == "digit3_alt" and (not goodDigit3)):
					goodDigitCond = self.goodDigitCondFuncDict[digName](digit)
				if digName.find("digit3") != -1 and goodDigitCond:
					goodDigit3 = True

				if digName.find("wellInd_dig") != -1:
					goodDigitCond = digit.sum() > misplacedOrNoDigitNorm

			else:

				if (digName == "digit3" or digName == "wellInd_dig2"):
					goodDigitCond = digit.sum() > blankDigitNorm
				if digName != "digit3" and digName != "wellInd_dig2":
					goodDigitCond = True

			if goodDigitCond:

				flatDigit = digit.reshape((-1))

				#This 3D tensor contains all the examples for all the class
				#for label in self.refDict[digName].keys():
				#	print(label,self.refDict[digName][label].shape)

				refTens = np.concatenate([self.refDict[digName][label][np.newaxis] for label in self.refDict[digName].keys()],axis=0)

				#Computing the average distance of the digits with examples of each class
				meanDist = np.sqrt(np.power(flatDigit-refTens,2).sum(axis=-1)).mean(axis=-1)
				ind = np.argmin(meanDist)

				label = sorted(glob.glob("../data/{}/timeImg/{}/*/".format(self.dataset,digName)))[ind].split("/")[-2]

				rawResDict[digName] = int(label)

			else:

				rawResDict[digName] = None
		#print(rawResDict)
		mergedResDict = {}
		if self.dataset == "big":
			#Merging all the key,value pairs corresponding to the same digit (i.e. mergin digit1 with digit1_alt and digit1_alt2)
			for dig in list(["digit1","digit2","digit3","wellInd_dig1","wellInd_dig2"]):

				digNames = sorted([digName for digName in pos.keys() if digName.find(dig) != -1])
				digFound = False
				i=0
				while i<len(digNames) and not digFound:
					if not rawResDict[digNames[i]] is None:
						digFound = True
						mergedResDict[dig] = rawResDict[digNames[i]]
					i+=1
				if not digFound:
					mergedResDict[dig] = None
			rawResDict = mergedResDict

		resDict = {}

		#Now we have all the digits, let's merge them

		#If the well index is inferior to 10, the second digit will be blank
		if rawResDict["wellInd_dig2"] is None:
			resDict["wellInd"] = rawResDict["wellInd_dig1"]
		else:
			resDict["wellInd"] =  10*rawResDict["wellInd_dig1"]+rawResDict["wellInd_dig2"]

		#If the time is inferior to 10, the third digit will be blank
		if rawResDict["digit3"] is None:
			resDict["time"] = rawResDict["digit2"]+0.1*rawResDict["digit1"]
		else:
			resDict["time"] = 10*rawResDict["digit3"]+rawResDict["digit2"]+0.1*rawResDict["digit1"]

		if not newVid:
			#To detect indirectly if there is a fourth digit (which is always equal to one),
			#we check if the current time is inferior to the last time.
			#As the clock cannot go back in time, current time < last time imply a fourth digit has appeared (which is alwayd a '1')
			if resDict["time"] < self.lastTime:
				resDict["time"] += 100

		self.lastTime = resDict["time"]

		return resDict

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Crop digits in the embryo videos and cluster them.')

	parser.add_argument('--img_nb', metavar='NB',help='The number of image to crop',type=int,default=2000)
	parser.add_argument('--dataset', metavar='DATASET',help='The dataset to extract digits from',type=str,default="small")

	args = parser.parse_args()

	clusterDigits(args.dataset,args.img_nb)
