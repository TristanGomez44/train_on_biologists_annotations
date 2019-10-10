import numpy as np
import cv2
import glob
import os
import argparse
from sklearn.cluster import KMeans
import numpy as np

#This dictionary indicates the relative position of the crop to do to obtain all the digits indicating the hour
imgHeigth,imgWidth = 500,500

heigth,width = 15,8

pos = {"digit1"  : {"Y1":475,"Y2":475+heigth,"X1":472,"X2":472+width}, \
	   "digit2" :  {"Y1":475,"Y2":475+heigth,"X1":462,"X2":462+width}, \
	   "digit3" :  {"Y1":475,"Y2":475+heigth,"X1":456,"X2":456+width}, \
	   "wellInd_dig1" : {"Y1":475,"Y2":475+heigth,"X1":37, "X2":37+width}, \
	   "wellInd_dig2" : {"Y1":475,"Y2":475+heigth,"X1":44, "X2":44+width} }

clustNb = {"digit1":10,"digit2":10,"digit3":10,"wellInd_dig1":9,"wellInd_dig2":3}

def clusterDigits(imgNb):
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
	vidPaths = glob.glob("../data/*avi")
	#The total nb of images decoded
	totalImgCount = 0
	vidInd = 0

	#The number of well id crop to extract per video
	imgNb_per_videos = imgNb//len(vidPaths)

	if not os.path.exists("../data/timeImg/"):
		os.makedirs("../data/timeImg/")

	for key in pos.keys():
		if not os.path.exists("../data/timeImg/{}/".format(key)):
			os.makedirs("../data/timeImg/{}/".format(key))

	while vidInd < len(vidPaths):
		#The number of images decoded in the current video
		videoImgCount = 0
		print(vidInd)
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
					if ((key == "digit3" or key == "wellInd_dig2") and digit.sum() > 30000) or (key != "digit3" and key != "wellInd_dig2"):
						cv2.imwrite("../data/timeImg/{}/{}.png".format(key,totalImgCount),digit)

			ret, frame = cap.read()
			totalImgCount += 1
			videoImgCount += 1

		vidInd += 1

	print("Clustering the digits ... ")
	for key in pos.keys():
		print("\t ",key)
		imgPaths = glob.glob("../data/timeImg/{}/*.png".format(key))
		data = []

		for path in imgPaths:
			img = cv2.imread(path)
			data.append(img.reshape((-1))[np.newaxis])

		data = np.concatenate(data,axis=0)
		kmeans = KMeans(n_clusters=clustNb[key], random_state=0).fit(data)

		for i,imgPath in enumerate(imgPaths):
			if not os.path.exists("../data/timeImg/{}/{}/".format(key,kmeans.labels_[i])):
				os.makedirs("../data/timeImg/{}/{}/".format(key,kmeans.labels_[i]))
			fileName = os.path.basename(imgPath)
			os.rename(imgPath,"../data/timeImg/{}/{}/{}".format(key,kmeans.labels_[i],fileName))

	print("Done ! Don't forget to rename the folders according to the true labels of the digit in it")


if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Crop digits in the embryo videos and cluster them.')

	parser.add_argument('--img_nb', metavar='PATH',help='The number of image to crop',type=str,default=2000)

	args = parser.parse_args()

	clusterDigits(args.img_nb)

	"""
	for key in pos.keys():
		vidPath = "../data/HH569 EMB2 FREEZE.avi"
		imgCount = 0
		vidInd = 0

		cap = cv2.VideoCapture(vidPath)
		ret, frame = cap.read()

		while ret:
			frameSize = frame.shape
			if imgCount == 500:
				cv2.imwrite("../data/test_{}.png".format(key),frame[pos[key]["Y1"]:pos[key]["Y2"],
													   	            pos[key]["X1"]:pos[key]["X2"],:])
				cv2.imwrite("../data/test_full.png",frame)
				break
			ret, frame = cap.read()
			imgCount += 1
	"""
