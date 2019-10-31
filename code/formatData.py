import glob
import os
import sys
import xlrd
import pandas as pd
import subprocess
import numpy as np
import digitExtractor
import cv2
import subprocess
from args import ArgReader

labelDict = {"tPB2":0,"tPNa":1,"tPNf":2,"t2":3,"t3":4,"t4":5,"t5":6,"t6":7,"t7":8,"t8":9,"t9+":10,"tM":11,"tSB":12,"tB":13,"tEB":14,"tHB":15}

def formatData(dataset,pathToZip,img_for_crop_nb):

    #Unziping and renaming the folder
    if dataset == "small":
        if not os.path.exists("../data/{}/".format(dataset)):
            subprocess.call("unzip {} -d ../data/".format(pathToZip),shell=True)
            subprocess.call("mv {} {}".format(os.path.splitext(pathToZip)[0],"../data/{}/".format(dataset)),shell=True)
    else:
        raise ValueError("Unkown dataset :",dataset)

    #Adding an underscore between the name of the video and the "EXPORT.xls" in the name of the excel files
    for xlsPath in glob.glob("../data/{}/*.xls".format(dataset)):
        #Checking if it has already been done before doing it so it does not add a second underscore
        if xlsPath.find("_EXPORT") == -1:
            newName = xlsPath[:xlsPath.find("EXPORT")]+"_EXPORT.xls"
            newName = newName.replace(" ","")
            os.rename(xlsPath,newName)

    #Removing space in video names
    for vidPath in glob.glob("../data/{}/*.avi".format(dataset)):
        os.rename(vidPath,vidPath.replace(" ","_"))

    #The folder than will contain the annotations
    if not os.path.exists("../data/{}/annotations/".format(dataset)):
        os.makedirs("../data/{}/annotations".format(dataset))

    #Convert the xls files into csv files if it is not already done
    if len(glob.glob("../data/{}/*.csv".format(dataset))) < len(glob.glob("../data/{}/*.xls".format(dataset))):
        subprocess.call("libreoffice --headless --convert-to csv --outdir ../data/{}/ ../data/{}/*.xls".format(dataset,dataset),shell=True)

    if not os.path.exists("../data/{}/timeImg/".format(dataset)):
        #Extracting and clustering the digits
        digitExtractor.clusterDigits(dataset,img_for_crop_nb)

    digExt = digitExtractor.DigitIdentifier(dataset)

    vidPaths=sorted(glob.glob("../data/{}/*avi".format(dataset)))

    for vidPath in vidPaths:
        print(vidPath)

        vidName = os.path.splitext(os.path.basename(vidPath))[0]

        if not os.path.exists("../data/{}/annotations/{}_phases.csv".format(dataset,vidName)):

            csvPath = os.path.dirname(vidPath) + "/"+ os.path.basename(vidPath).split("_")[0] + "_EXPORT.csv"
            df = pd.read_csv(csvPath)[["Well"]+list(labelDict.keys())]

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
                line = df.loc[df['Well'] == wellInd][list(labelDict.keys())]

                for col in line.columns:
                    line[col] = line[col].apply(lambda x:x.replace(",",".") if type(x) == str else x).astype(float)

                #Removes label columns that do not appear in the video (i.e. those with NaN value)
                line = line.transpose()
                line = line[np.isnan(line) == 0]
                line = line.transpose()

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

            #Adding the last phase
            labDict[currentLabel] = (startOfCurrentPhase,imgCount-1)

            #Writing the start and end frames of each phase in a csv file
            with open("../data/{}/annotations/{}_phases.csv".format(dataset,vidName),"w") as text_file:
                for label in labelDict.keys():
                    if label in labDict.keys():
                        print(label+","+str(labDict[label][0])+","+str(labDict[label][1]),file=text_file)

def getLabels():
    return labelDict

def getReversedLabels():

    revDict = {}
    for key in labelDict.keys():
        revDict[labelDict[key]] = key

    return revDict

def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)

    ########### PLOT SCORE EVOLUTION ALONG VIDEO ##################
    argreader.parser.add_argument('--dataset',type=str,metavar="DATASET",help='The dataset to format')
    argreader.parser.add_argument('--path_to_zip',type=str,metavar="PATH",help='The path to the zip file containing the dataset "small". Only used if it is the dataset "small" that is \
                                     being formated.',default="../data/embryo.zip")
    argreader.parser.add_argument('--img_for_crop_nb',type=int,metavar="NB",help='The number of images from which to extract digits',default=2000)

    #Reading the comand line arg
    argreader.getRemainingArgs()

    #Getting the args from command line and config file
    args = argreader.args

    formatData(args.dataset,args.path_to_zip,args.img_for_crop_nb)

if __name__ == "__main__":
    main()
