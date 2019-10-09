import numpy as np
import cv2
import glob
import os
import argparse 
from sklearn.cluster import KMeans

if __name__ == "__main__":

	parser = argparse.ArgumentParser(description='Crop digits in the embryo videos and cluster them.')

	parser.add_argument('--img_nb', metavar='PATH',help='The number of image to crop',type=str,default=2000)

	args = parser.parse_args()

	imgNb = args.img_nb

	print("Extracting digit from the images...")
	vidPaths = glob.glob("../data/*avi")
	imgCount = 0
	vidInd = 0

	if not os.path.exists("../data/timeImg/"):
		os.makedirs("../data/timeImg/")

	while vidInd < len(vidPaths) and imgCount < imgNb:
		
		cap = cv2.VideoCapture(vidPaths[vidInd])
		ret, frame = cap.read()
		while ret and imgCount < imgNb:
			
			
			frameSize = frame.shape
		
			cv2.imwrite("../data/timeImg/{}.png".format(imgCount),frame[int(frameSize[0]*0.95):int(frameSize[0]*0.98),int(frameSize[1]*0.945):int(frameSize[0]*0.965),:])
			ret, frame = cap.read()				
			imgCount += 1
		
		vidInd += 1

	print("Clustering the digits ... ")	
	imgPaths = glob.glob("../data/timeImg/*.png")
	data = []
	
	for path in imgPaths:
		img = cv2.imread(path)	
		data.append(img.reshape((-1))[np.newaxis])

	data = np.concatenate(data,axis=0)
	kmeans = KMeans(n_clusters=10, random_state=0).fit(data)
	
	for i,imgPath in enumerate(imgPaths):
		if not os.path.exists("../data/timeImg/{}/".format(kmeans.labels_[i])):
			os.makedirs("../data/timeImg/{}/".format(kmeans.labels_[i]))
		fileName = os.path.basename(imgPath)
		os.rename(imgPath,"../data/timeImg/{}/{}".format(kmeans.labels_[i],fileName))

	print("Done ! Don't forget to rename the folders according to the true labels of the digit in it")
	
