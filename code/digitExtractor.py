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

clustNb = {"digit1":10,"digit2":10,"digit3":10,"wellInd_dig1":9,"wellInd_dig2":3}

#This dict map cluster index to the real class (i.e. the digit)
clustInd2Digit = {"digit1":{0:9, 1:1, 2:7, 3:6, 4:8, 5:2, 6:4, 7:0, 8:5, 9:3},
				  "digit2":{0:9, 1:4, 2:3, 3:5, 4:7, 5:1, 6:2, 7:8, 8:0, 9:6},
				  "digit3":{0:4, 1:3, 2:5, 3:1, 4:7, 5:2, 6:0, 7:8, 8:9, 9:6},
				  "wellInd_dig1":{0:8, 1:1, 2:2, 3:5, 4:7, 5:4, 6:9, 7:6, 8:3},
				  "wellInd_dig2":{0:0, 1:1, 2:2}}

blankDigitNorm = 30000

def getDigitYPos():
	return pos["digit1"]["Y1"],pos["digit1"]["Y2"]

def clusterDigits(dataset,imgNb):
	"""	Extract digits from embryo images in the ../data/ folder and cluster them using k-means

	Once it is finished running, in ../data/ a folder "timeImg" is created and contains 4 folders : digit1, digit2, digit3, digit4, and wellInd.
	Each folder is dedicated to one digit visible in the image. It can be the well index (bottom left of the image), or one of the four digits
	indicating the hour at which the image was taken (bottom right). In each of those 4 folder you will find 10 folders, each containing a cluster
	of digits. You have to rename manually those folders with the correct class. E.g., the folder containing all the "1" should be renamed "1".

	Args:
		- imgNb (int) : the number of digit crops to extract. For the digits indicating the time, they will be extracted from all the frame of the first videos and for the digit
						indicating the well id, only the first few frames of all the videos will be extracted.

	"""

	print("Extracting digit from the images...")
	vidPaths = glob.glob("../data/{}/*avi".format(dataset))
	#The total nb of images decoded
	totalImgCount = 0
	vidInd = 0

	#The number of well id crop to extract per video
	imgNb_per_videos = imgNb//len(vidPaths)

	if not os.path.exists("../data/{}/timeImg/".format(dataset)):
		os.makedirs("../data/{}/timeImg/".format(dataset))

	for key in pos.keys():
		if not os.path.exists("../data/{}/timeImg/{}/".format(dataset,key)):
			os.makedirs("../data/{}/timeImg/{}/".format(dataset,key))

	while vidInd < len(vidPaths):
		#The number of images decoded in the current video
		videoImgCount = 0

		cap = cv2.VideoCapture(vidPaths[vidInd])
		ret, frame = cap.read()
		while ret:

			frameSize = frame.shape

			for key in pos.keys():
				digit = frame[pos[key]["Y1"]:pos[key]["Y2"],pos[key]["X1"]:pos[key]["X2"],:]

				#For the crop of the well id, only the first few frame of each video are extracted
				#For the crop of the time digit, all the frames of the first videos are extracted
				if (key.find("wellInd") != -1 and videoImgCount < imgNb_per_videos) or (key.find("digit") != -1 and totalImgCount < imgNb):
					#If the crop is blank (because there is not digit yet at this position in the video), it is not necessary to write it
					if ((key == "digit3" or key == "wellInd_dig2") and digit.sum() > blankDigitNorm) or (key != "digit3" and key != "wellInd_dig2"):
						cv2.imwrite("../data/{}/timeImg/{}/{}.png".format(dataset,key,totalImgCount),digit)

			ret, frame = cap.read()
			totalImgCount += 1
			videoImgCount += 1

		vidInd += 1

	print("Clustering the digits ... ")
	for key in pos.keys():
		imgPaths = glob.glob("../data/{}/timeImg/{}/*.png".format(dataset,key))
		data = []

		for path in imgPaths:
			img = cv2.imread(path)
			data.append(img.reshape((-1))[np.newaxis])

		data = np.concatenate(data,axis=0)
		kmeans = KMeans(n_clusters=clustNb[key], random_state=0).fit(data)

		#Moving the image in their respective folder
		for i,imgPath in enumerate(imgPaths):
			if not os.path.exists("../data/{}/timeImg/{}/{}/".format(dataset,key,kmeans.labels_[i])):
				os.makedirs("../data/{}/timeImg/{}/{}/".format(dataset,key,kmeans.labels_[i]))
			fileName = os.path.basename(imgPath)
			os.rename(imgPath,"../data/{}/timeImg/{}/{}/{}".format(dataset,key,kmeans.labels_[i],fileName))

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
		for digName in pos.keys():

			self.refDict[digName] = {}

			for label in sorted(glob.glob("../data/{}/timeImg/{}/*/".format(dataset,digName))):

				#This is a 2D matrix where each row is one example of the class
				imgList = np.concatenate(list(map(lambda x:cv2.imread(x).reshape((-1))[np.newaxis],sorted(glob.glob(label+"/*.png".format(digName)))[:neigbhorsNb])),axis=0)
				self.refDict[digName][label] = imgList

	def findDigits(self,img):
		''' Crop the digits on an image and identify them with KNN algorithm

		The image contains two information  :
		- The well index, on the bottom left
		- The time at which the photo was taken, on the bottom right

		Args:
		- img (3D array): the image
		Returns:
		- resDict (dict): a dictionary containing two keys :
			- "wellInd" : the index of the well
			- "time" : the time at which the image was taken

		'''

		rawResDict = {}
		for digName in pos.keys():
			digit = img[pos[digName]["Y1"]:pos[digName]["Y2"],pos[digName]["X1"]:pos[digName]["X2"],:]
			flatDigit = digit.reshape((-1))

			#This 3D tensor contains all the examples for all the class
			refTens = np.concatenate([self.refDict[digName][label][np.newaxis] for label in self.refDict[digName].keys()],axis=0)

			#Computing the average distance of the digits with examples of each class
			meanDist = np.sqrt(np.power(flatDigit-refTens,2).sum(axis=-1)).mean(axis=-1)
			ind = np.argmin(meanDist)

			label = sorted(glob.glob("../data/{}/timeImg/{}/*/".format(self.dataset,digName)))[ind].split("/")[-2]

			rawResDict[digName] = int(label) if digit.sum() >= blankDigitNorm else None

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
			#If the time is inferior to 100, the fourth digit will be blank
			if img[hundredsDigitPos["Y1"]:hundredsDigitPos["Y2"],hundredsDigitPos["X1"]:hundredsDigitPos["X2"],:].sum() < blankDigitNorm:
				resDict["time"] = 10*rawResDict["digit3"]+rawResDict["digit2"]+0.1*rawResDict["digit1"]
			else:
				resDict["time"] = 100+10*rawResDict["digit3"]+rawResDict["digit2"]+0.1*rawResDict["digit1"]

		return resDict

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Crop digits in the embryo videos and cluster them.')

	parser.add_argument('--img_nb', metavar='NB',help='The number of image to crop',type=int,default=2000)
	parser.add_argument('--dataset', metavar='DATASET',help='The dataset to extract digits from',type=str,default="small")

	args = parser.parse_args()

	clusterDigits(args.dataset,args.img_nb)
