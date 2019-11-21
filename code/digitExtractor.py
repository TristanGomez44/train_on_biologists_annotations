import numpy as np
import cv2
import glob
import os
import argparse
from sklearn.cluster import KMeans
import numpy as np
import sys
from scipy import ndimage

imgHeigth = 500

def getVideosToRemove():
	return ["BE150-1","MZ900-8","NE051-1","ZVDPI098SLIDE2-3","MS288SLIDE1-1","FE14-010","RA444-5"]

def clusterDigits(dataset,imgNb):
	"""	Extract digits from embryo images in the ../data/ folder and cluster them using k-means

	Once it is finished running, in ../data/ a folder "timeImg" is created and contains several folders : digit1, digit2, digit3, wellInd, etc.
	Each folder is dedicated to one digit visible in the image. It can be the well index (bottom left of the image), or one of the four digits
	indicating the hour at which the image was taken (bottom right). In each of those 4 folder you will find 10 folders, each containing a cluster
	of digits. You have to rename manually those folders with the correct class. E.g., the folder containing all the "1" should be renamed "1".

	Args:
		- imgNb (int) : the number of digit crops to extract. For the digits indicating the time, they will be extracted from all the frame of the first videos and for the digit
						indicating the well id, only the first few frames of all the videos will be extracted.

	"""

	if dataset == "big":

		clustNb = {"digit1_14_3":1,"digit1_14_4":6 ,"digit1_14_5":3,"digit1_15_4":1,"digit1_15_5":9,
				   "digit2_14_3":1,"digit2_14_4":8,"digit2_14_5":3,"digit2_15_4":1,"digit2_15_5":9,
				   "digit3_14_3":1,"digit3_14_4":6,"digit3_14_5":3,"digit3_14_6":1,"digit3_15_4":1,"digit3_15_5":9,"digit3_15_6":1,
				   "digit4_14_3":1,"digit4_15_4":1,
				   "wellInd_dig1_14_3":1,"wellInd_dig1_14_4":6,"wellInd_dig1_14_5":3,"wellInd_dig1_15_4":1,"wellInd_dig1_15_5":9,
				   "wellInd_dig2_14_3":1,"wellInd_dig2_14_4":1,"wellInd_dig2_15_4":1}

		#This dict map cluster index to the real class (i.e. the digit)
		clustInd2Digit = {"digit1_14_3":{0:1},
						  "digit1_14_4":{0:7,1:0,2:4,3:5,4:3,5:9},
						  "digit1_14_5":{0:6,1:8,2:2},
						  "digit1_15_4":{0:1},
						  "digit1_15_5":{0:7,1:0,2:5,3:4,4:2,5:3,6:8,7:6,8:9},
						  "digit2_14_3":{0:1},
						  "digit2_14_4":{0:5,1:4,2:7,3:9,4:3,5:0,6:2,7:6},
						  "digit2_14_5":{0:6,1:2,2:8},
						  "digit2_15_4":{0:1},
						  "digit2_15_5":{0:6,1:2,2:4,3:8,4:7,5:0,6:3,7:9,8:5},
						  "digit3_14_3":{0:1},
						  "digit3_14_4":{0:4,1:3,2:9,3:7,4:0,5:5},
						  "digit3_14_5":{0:6,1:2,2:8},
						  "digit3_14_6":{0:8},
						  "digit3_15_4":{0:1},
						  "digit3_15_5":{0:0,1:2,2:7,3:5,4:3,5:4,6:8,7:6,8:9},
						  "digit3_15_6":{0:4},
						  "digit4_14_3":{0:1},
						  "digit4_15_4":{0:1},
						  "wellInd_dig1_14_3":{0:1},
						  "wellInd_dig1_14_4":{0:0,1:3,2:4,3:7,4:5,5:9},
						  "wellInd_dig1_14_5":{0:8,1:6,2:2},
						  "wellInd_dig1_15_4":{0:1},
						  "wellInd_dig1_15_5":{0:2,1:8,2:4,3:3,4:5,5:7,6:6,7:9,8:0},
						  "wellInd_dig2_14_3":{0:1},
						  "wellInd_dig2_14_4":{0:0},
						  "wellInd_dig2_15_4":{0:1}}

	else:

		clustNb = {"digit1_15_4":1,"digit1_15_5":9,
				   "digit2_15_4":1,"digit2_15_5":9,
				   "digit3_15_4":1,"digit3_15_5":9,"digit3_15_6":1,
				   "digit4_15_4":1,
				   "wellInd_dig1_15_4":1,"wellInd_dig1_15_5":9,
				   "wellInd_dig2_15_4":1}

		#This dict map cluster index to the real class (i.e. the digit)
		clustInd2Digit = {"digit1_15_4":{0:1},
						  "digit1_15_5":{0:0,1:4,2:7,3:2,4:6,5:8,6:9,7:5,8:3},
						  "digit2_15_4":{0:1},
						  "digit2_15_5":{0:4,1:0,2:7,3:2,4:5,5:3,6:8,7:9,8:6},
						  "digit3_15_4":{0:1},
						  "digit3_15_5":{0:2,1:0,2:3,3:7,4:6,5:4,6:8,7:9,8:5},
						  "digit3_15_6":{0:4},
						  "digit4_15_4":{0:1},
						  "wellInd_dig1_15_4":{0:1},
						  "wellInd_dig1_15_5":{0:3,1:4,2:6,3:7,4:2,5:0,6:8,7:5,8:9},
						  "wellInd_dig2_15_4":{0:1}}

	vidPaths = glob.glob("../data/{}/*avi".format(dataset))

	if not os.path.exists("../data/{}/timeImg/".format(dataset)):
		print("Extracting digit from the images...")

		if dataset == "big":
			#Removing the videos with big problems
			for vidName in getVideosToRemove():
				vidPaths.remove("../data/{}/".format(dataset)+vidName+".avi")

		vidInd = 0

		#Store the total number of cropped images written for each digit
		totalCountDict = {}
		for key in getKeys(dataset):
			totalCountDict[key] = 0

		if not os.path.exists("../data/{}/timeImg/".format(dataset)):
			os.makedirs("../data/{}/timeImg/".format(dataset))

		wellIndCropPerVid = 1 + imgNb//len(vidPaths)

		while vidInd < len(vidPaths):
			#The number of images decoded in the current video
			videoImgCount = 0
			print(vidInd,"/",len(vidPaths)," : ",vidPaths[vidInd])
			cap = cv2.VideoCapture(vidPaths[vidInd])

			ret, frame = cap.read()
			frameInd = 0
			enoughImgParsed = False

			while ret and not enoughImgParsed:
				frame = frame[frame.shape[0]-imgHeigth:,:]
				frameInd += 1

				if frameInd == 1:
					processFrame(frame,dataset,totalCountDict,imgNb,extractWellId=True)
				processFrame(frame,dataset,totalCountDict,imgNb,extractWellId=False)

				enoughImgParsed = sum([totalCountDict[digitName]==imgNb for digitName in totalCountDict.keys() if digitName.find("digit") != -1]) == len([digitName for digitName in totalCountDict.keys() if digitName.find("digit") != -1])

				ret, frame = cap.read()

			vidInd += 1

	print("Clustering the digits ... ")
	for key in getKeys(dataset):

		imgPaths = sorted(glob.glob("../data/{}/timeImg/{}/*.png".format(dataset,key)))
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

	print("Renaming the folder according to the digits they contain")
	for key in getKeys(dataset):

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

def getKeys(dataset,wellId=None,cannonical=False):

	if not cannonical:
		allKeys = list(map(lambda x:x.split("/")[-2],sorted(glob.glob("../data/{}/timeImg/*/".format(dataset)))))

		if wellId is None:
			return allKeys
		elif wellId == True:
			return [key for key in allKeys if key.find("wellId") != -1]
		else:
			return [key for key in allKeys if key.find("wellId") == -1]
	else:

		if wellId is None:
			return ["digit1","digit2","digit3","digit4","wellInd_dig1","wellInd_dig2"]
		elif wellId == True:
			return ["wellInd_dig1","wellInd_dig2"]
		else:
			return ["digit1","digit2","digit3","digit4"]

def computeGradMag(img):
	img = img.astype('int32')
	dx = ndimage.sobel(img, 0)  # horizontal derivative
	dy = ndimage.sobel(img, 1)  # vertical derivative
	mag = np.hypot(dx, dy)  # magnitude
	#mag = dx.astype("float64")
	mag *= 255.0 / np.max(mag)  # normalize (Q&D)
	mag= mag.astype("uint8")
	return mag

def processFrame(img,dataset,totalCountDict=None,requiredImgNb=None,extractWellId=False,write=True):

	if not extractWellId:
		img = (img[-50:,-50:].mean(axis=-1)).astype("uint8")
	else:
		img = (img[-50:,:70].mean(axis=-1)).astype("uint8")

	imgBin = img>128

	rightleft = imgBin.sum(axis=1)>2
	topbottom = imgBin.sum(axis=0)>2
	x,y = rightleft.argmax(),topbottom.argmax()
	width,heigth = rightleft.sum(),topbottom.sum()

	if width > 20:
		gradMag = computeGradMag(img)
		cv2.imwrite("../data/big/timeImg/preGrad.png",img)

		img[(gradMag<100)*(img<180)] = 0

		cv2.imwrite("../data/big/timeImg/postGrad.png",img)
		cv2.imwrite("../data/big/timeImg/grad.png",gradMag)
		imgBin = img>128
		rightleft = imgBin.sum(axis=1)>2
		topbottom = imgBin.sum(axis=0)>2
		x,y = rightleft.argmax(),topbottom.argmax()
		width,heigth = rightleft.sum(),topbottom.sum()

	img = img[x:width+x,y:y+heigth]
	cv2.imwrite("../data/big/timeImg/crop.png",img)
	projX = (img<128).sum(axis=0)

	offSet = 0
	endReached = False
	digitList = []
	while not endReached:

		start = (projX[offSet:] != 0).argmax()+offSet
		length = min((projX[start:] != 0).argmin(),6)

		end = start+length

		if start==end:
			endReached=True
		else:
			digitList.append(img[:,start:end])
			offSet = end

	if not extractWellId:
		#Removing the character 'h' and the comma
		digitList.pop(-1)
		digitList.pop(-2)
	else:
		#Removing the characters of the word "Well"
		digitList = digitList[4:]

	resDict = {}

	for i,dig in enumerate(digitList):

		if dig.shape[1] > 1:

			if not extractWellId:
				digitName = "digit"+str(len(digitList)-i)+"_"+str(dig.shape[0])+"_"+str(dig.shape[1])
			else:
				digitName = "wellInd_dig"+str(len(digitList)-i)+"_"+str(dig.shape[0])+"_"+str(dig.shape[1])

			if write:
				if digitName in totalCountDict.keys():
					if totalCountDict[digitName]<requiredImgNb:
						cv2.imwrite("../data/{}/timeImg/{}/{}.png".format(dataset,digitName,totalCountDict[digitName]),dig)
						totalCountDict[digitName] += 1
				else:
					totalCountDict[digitName] = 1
					os.makedirs("../data/{}/timeImg/{}/".format(dataset,digitName))
					cv2.imwrite("../data/{}/timeImg/{}/{}.png".format(dataset,digitName,totalCountDict[digitName]),dig)

			resDict[digitName] = dig

	return resDict

def computeRealFrameRate(vidPaths,dataset,neigbhorsNb):

	digitIdentif = DigitIdentifier(dataset,neigbhorsNb)

	realFrameRateCsv = 'video_name,real_frame_rate\n'

	for vidInd in range(len(vidPaths)):

		if vidInd%10==0:
			print(vidPaths[vidInd])

		cap = cv2.VideoCapture(vidPaths[vidInd])
		ret, frame = cap.read()
		frameInd = 0

		resDict = digitIdentif.findDigits(frame)
		lastTime = resDict["time"]
		lastTimeFrameInd = 0
		startTime = resDict["time"]

		enoughImgParsed	= False

		while ret and not enoughImgParsed:

			#If the current time is now inferior to the preceding one, it means it has got above 99,
			#(as we only check the three first digits) so we should stop
			if resDict["time"] < lastTime:
				enoughImgParsed = True
			else:
				resDict = digitIdentif.findDigits(frame)
				lastTime = resDict["time"]
				lastTimeFrameInd = frameInd
				print(lastTime,frameInd)
			#Fast forward in the video
			for _ in range(10):
				if ret:
					frameInd += 1
					ret, frame = cap.read()

		endTime = lastTime

		realFrameRate = (endTime - startTime)/(frameInd+1)

		realFrameRateCsv += os.path.splitext(os.path.basename(vidPaths[vidInd]))[0]+","+str(realFrameRate)+"\n"

		sys.exit(0)

	with open("../data/{}/annotations/realFrameRate.csv".format(dataset),"w") as text_file:
		print(realFrameRateCsv,file=text_file)

class DigitIdentifier:
	""" This class extracts the digits of an image

	First the digits on the images are cropped and then they identified with the KNN algorithm.
	Each crop is compared to 10 digits of each class and is assigned to the group from which it is the closest

	Args:
	- neigbhorsNb (int): the number of neigbhors to compare with
	- dataset (str): the dataset to process
	"""

	def __init__(self,dataset,neigbhorsNb=10):


		super(DigitIdentifier,self).__init__()

		self.refDict = {}
		self.dataset = dataset

		for digName in getKeys(self.dataset):

			self.refDict[digName] = {}

			minSize = np.inf
			for label in sorted(glob.glob("../data/{}/timeImg/{}/*/".format(dataset,digName))):
				#This is a 2D matrix where each row is one example of the class
				imgList = list(map(lambda x:cv2.imread(x),sorted(glob.glob(label+"/*.png".format(digName)))))

				data = []

				for img in imgList:

					data.append(img.reshape((-1))[np.newaxis])

				self.refDict[digName][label] =  np.concatenate(data,axis=0)

				#There are some labels for which there is fewer examples than other labels
				#It is necessary to select x examples for each labels, where x is the minimum number of examples for any label
				if len(imgList) < minSize:
					minSize = len(imgList)
			for label in sorted(glob.glob("../data/{}/timeImg/{}/*/".format(dataset,digName))):
				self.refDict[digName][label] = self.refDict[digName][label][:minSize]

	def findDigits(self,img,newVid=False,debug=False,extractWellId=False):
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

		timeRawResDict = processFrame(img,self.dataset,None,None,extractWellId=False,write=False)
		timeRawResDict = processDict(timeRawResDict,self.dataset,self.refDict)
		#print(timeRawResDict)
		timeRawResDict = mergeDict(timeRawResDict,self.dataset,extractWellId=False)

		if extractWellId:
			wellIndrawResDict = processFrame(img,self.dataset,None,None,extractWellId=True,write=False)
			wellIndrawResDict = processDict(wellIndrawResDict,self.dataset,self.refDict)
			wellIndrawResDict = mergeDict(wellIndrawResDict,self.dataset,extractWellId=True)

		resDict = {}

		#Now we have all the digits, let's merge them
		if extractWellId:

			#If the well index is inferior to 10, the second digit will be blank
			resDict["wellInd"] = wellIndrawResDict["wellInd_dig1"]

			if "wellInd_dig2" in list(wellIndrawResDict.keys()):
				resDict["wellInd"] +=  10*wellIndrawResDict["wellInd_dig2"]

		resDict["time"] = timeRawResDict["digit2"]+0.1*timeRawResDict["digit1"]

		if "digit3" in list(timeRawResDict.keys()):
			resDict["time"] += 10*timeRawResDict["digit3"]

		if "digit4" in list(timeRawResDict.keys()):
			resDict["time"] +=  100*timeRawResDict["digit4"]

		return resDict

def mergeDict(rawResDict,dataset,extractWellId):
	mergedResDict = {}
	#Merging all the key,value pairs corresponding to the same digit (i.e. mergin digit1 with digit1_alt and digit1_alt2)
	for dig in getKeys(dataset,extractWellId,cannonical=True):

		digNames = sorted([digName for digName in rawResDict.keys() if digName.find(dig) != -1])
		digFound = False
		i=0
		while i<len(digNames) and not digFound:
			if not rawResDict[digNames[i]] is None:
				digFound = True
				mergedResDict[dig] = rawResDict[digNames[i]]
			i+=1

	return mergedResDict

def processDict(rawResDict,dataset,refDict):

	for digName in rawResDict.keys():

		img = rawResDict[digName]

		img = img[:,:,np.newaxis].repeat(3,axis=-1)

		flatDigit = img.reshape((-1))

		#This 3D tensor contains all the examples for all the class
		refTens = np.concatenate([refDict[digName][label][np.newaxis] for label in refDict[digName].keys()],axis=0)

		#Computing the average distance of the digits with examples of each class
		meanDist = np.sqrt(np.power(flatDigit-refTens,2).sum(axis=-1)).mean(axis=-1)
		ind = np.argmin(meanDist)

		label = sorted(glob.glob("../data/{}/timeImg/{}/*/".format(dataset,digName)))[ind].split("/")[-2]

		rawResDict[digName] = int(label)

	return rawResDict

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Crop digits in the embryo videos and cluster them.')

	parser.add_argument('--img_nb', metavar='NB',help='The number of image to crop',type=int,default=2000)
	parser.add_argument('--dataset', metavar='DATASET',help='The dataset to extract digits from',type=str,default="small")

	args = parser.parse_args()

	clusterDigits(args.dataset,args.img_nb)
