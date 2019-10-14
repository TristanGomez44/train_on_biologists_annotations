import glob
import os
import sys
import xlrd
import pandas as pd
import subprocess
import numpy as np
import digitExtractor
import cv2

labels = ["tPB2","tPNa","tPNf","t2","t3","t4","t5","t6","t7","t8","t9+","tM","tSB","tB","tEB","tHB"]

def formatData():

    vidPaths=sorted(glob.glob("../data/*avi"))

    #Adding an underscore between the name of the video and the "EXPORT.xls" in the name of the excel files
    for xlsPath in glob.glob("../data/*.xls"):
        #Checking if it has already been done before doing it so it does not add a second underscore
        if xlsPath.find("_EXPORT") == -1:
            newName = xlsPath[:xlsPath.find("EXPORT")]+"_EXPORT.xls"
            os.rename(xlsPath,newName)

    #Removing space in video names
    for vidPath in glob.glob("../data/*.avi"):
        os.rename(vidPath,vidPath.replace(" ","_"))

    #The folder than will contain the annotations
    if not os.path.exists("../data/annotations/"):
        os.makedirs("../data/annotations")

    #Convert the xls files into csv files if it is not already done
    if len(glob.glob("../data/*.csv")) < len(glob.glob("../data/*.xls")):
        subprocess.call("libreoffice --headless --convert-to csv --outdir ../data/ ../data/*.xls",shell=True)

    digExt = digitExtractor.DigitIdentifier()

    badlyPositionedImageNb = 0
    badlyPositionedImageVidNames = []
    totalImgNb = 0

    for vidPath in vidPaths:
        print(vidPath)

        vidName = os.path.splitext(os.path.basename(vidPath))[0]

        if not os.path.exists("../data/annotations/{}_phases.csv".format(vidName)):

            csvPath = os.path.dirname(vidPath) + "/"+ os.path.basename(vidPath).split("_")[0] + "_EXPORT.csv"

            if not os.path.exists(csvPath):
                raise OSError("Missing csv path for {}".format(vidPath))

            df = pd.read_csv(csvPath)[["Well"]+labels]

            labDict = {}
            imgCount = 0
            startOfCurrentPhase = 0
            currentLabel = None

            #Reading the video
            cap = cv2.VideoCapture(vidPath)
            ret, frame = cap.read()
            resDict = digExt.findDigits(frame)
            wellInd = resDict["wellInd"]

            while ret:

                #Select only the label columns of the well
                line = df.loc[df['Well'] == wellInd][labels]

                #Removes label columns that do not appear in the video (i.e. those with NaN value)
                line = line.transpose()
                line = line[np.isnan(line[line.columns[0]]) == 0]
                line = line.transpose()

                #Counting the number of badly positionned images
                #If the well index found on the image does not appear in the excel sheet,
                #or is different from the index found in the begining of the video
                #it means the well index found is wrong and therefore the image is baddly positionned
                lineTest = df.loc[df['Well'] ==  resDict["wellInd"]][labels]
                if lineTest.empty:
                    badlyPositionedImageNb += 1
                    if not vidName in badlyPositionedImageVidNames:
                        badlyPositionedImageVidNames.append(vidName)

                #Getting the true label of the image
                label = line.columns[max((resDict["time"] > line).sum(axis=1).item()-1,0)]

                #Initialise currentLabel with the first label
                if currentLabel is None:
                    currentLabel = label

                #If this condition is true, the current frame belongs to a new phase
                if label != currentLabel:
                    #Adding the start and end frames of last phase in the dict
                    labDict[currentLabel] = (startOfCurrentPhase,imgCount-1)
                    startOfCurrentPhase = imgCount
                    currentLabel = label

                ret, frame = cap.read()
                if not frame is None:
                    resDict = digExt.findDigits(frame)

                imgCount +=1

            totalImgNb += imgCount

            #Adding the last phase
            labDict[currentLabel] = (startOfCurrentPhase,imgCount-1)

            #Writing the start and end frames of each phase in a csv file
            with open("../data/annotations/{}_phases.csv".format(vidName),"w") as text_file:
                for label in labels:
                    if label in labDict.keys():
                        print(label+","+str(labDict[label][0])+","+str(labDict[label][1]),file=text_file)

    print("proportion of badly positioned images : ",badlyPositionedImageNb/totalImgNb, "in videos : ",badlyPositionedImageVidNames)

if __name__ == "__main__":
    formatData()
