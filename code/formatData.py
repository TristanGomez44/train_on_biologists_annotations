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
from shutil import copyfile

labelDict = {"tPB2":0,"tPNa":1,"tPNf":2,"t2":3,"t3":4,"t4":5,"t5":6,"t6":7,"t7":8,"t8":9,"t9+":10,"tM":11,"tSB":12,"tB":13,"tEB":14,"tHB":15}

def formatDataSmall(dataset,pathToZip,img_for_crop_nb):

    #Unziping and renaming the folder
    if not os.path.exists("../data/{}/".format(dataset)):
        subprocess.call("unzip {} -d ../data/".format(pathToZip),shell=True)
        subprocess.call("mv {} {}".format(os.path.splitext(pathToZip)[0],"../data/{}/".format(dataset)),shell=True)

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

            phaseDict = {}
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
                row = df.loc[df['Well'] == wellInd][list(labelDict.keys())]

                for col in row.columns:
                    row[col] = row[col].apply(lambda x:x.replace(",",".") if type(x) == str else x).astype(float)

                #Removes label columns that do not appear in the video (i.e. those with NaN value)
                row = row.transpose()
                row = row[np.isnan(row) == 0]
                row = row.transpose()

                #Getting the true label of the image
                label = row.columns[max((resDict["time"] > row).sum(axis=1).item()-1,0)]

                #Initialise currentLabel with the first label
                if currentLabel is None:
                    currentLabel = label

                #If this condition is true, the current frame belongs to a new phase
                if label != currentLabel:
                    #Adding the start and end frames of last phase in the dict
                    phaseDict[currentLabel] = (startOfCurrentPhase,imgCount-1)
                    startOfCurrentPhase = imgCount
                    currentLabel = label

                ret, frame = cap.read()
                if not frame is None:
                    resDict = digExt.findDigits(frame)

                imgCount +=1

            #Adding the last phase
            phaseDict[currentLabel] = (startOfCurrentPhase,imgCount-1)

            #Writing the start and end frames of each phase in a csv file
            with open("../data/{}/annotations/{}_phases.csv".format(dataset,vidName),"w") as text_file:
                for label in labelDict.keys():
                    if label in phaseDict.keys():
                        print(label+","+str(phaseDict[label][0])+","+str(phaseDict[label][1]),file=text_file)

def formatDataBig(dataset,pathToFold,img_for_crop_nb):

    if not os.path.exists("../data/{}/".format(dataset)):
        os.makedirs("../data/{}/".format(dataset))

    #Moving the videos
    for videoPath in sorted(glob.glob(pathToFold+"/DATA/*avi")):
        os.rename(videoPath,"../data/{}/".format(dataset)+os.path.basename(videoPath))

    #Moving the excel files
    for excelPath in sorted(glob.glob(pathToFold+"/*.xls*")):
        os.rename(excelPath,"../data/{}/".format(dataset)+os.path.basename(excelPath))

    #Moving the csv file
    copyfile(pathToFold+"/AnnotationManuelle2017.csv","../data/{}".format(dataset)+"/AnnotationManuelle2017.csv")

    #Adding an underscore between the name of the video and the "EXPORT.xls" in the name of the excel files
    for xlsPath in glob.glob("../data/{}/*.xls".format(dataset)):

        dirName = os.path.dirname(xlsPath)
        newFileName = os.path.basename(xlsPath).replace(" ","_")
        os.rename(xlsPath,dirName+"/"+newFileName)

    #Convert the xls files into csv files if it is not already done
    if (len(glob.glob("../data/{}/*.csv".format(dataset))) - 1) < len(glob.glob("../data/{}/*.xls*".format(dataset))):
        subprocess.call("libreoffice --headless --convert-to csv --outdir ../data/{}/ ../data/{}/*.xls*".format(dataset,dataset),shell=True)

    #The folder than will contain the annotations
    if not os.path.exists("../data/{}/annotations/".format(dataset)):
        os.makedirs("../data/{}/annotations".format(dataset))

    def preproc(x):

        x = x.replace("/","").replace("DPI","").replace(" ","")

        i=0
        endOfNumber = False
        startOfNumber = False

        while not endOfNumber and i<len(x):

            if x[i].isdigit():
                if not startOfNumber:
                    startOfNumber = True
            else:
                if startOfNumber:
                    endOfNumber = True

            i+=1

        x = x[:i-1] if endOfNumber else x

        return x

    dfDict = {}
    for csvPath in sorted(glob.glob("../data/{}/*.csv".format(dataset))):

        if os.path.basename(csvPath) == "annoted31.12.2016.csv" or os.path.basename(csvPath) == "export_18-05-16.csv" or os.path.basename(csvPath) == "ALR493_EXPORT.csv" or os.path.basename(csvPath) == "DC307_EXPORT.csv":
            df = pd.read_csv(csvPath,dtype=str,encoding = "ISO-8859-1",sep=",")
            idColName = "PatientName"

            names = df[idColName].apply(preproc)
            df = df[["Well"]+list(labelDict.keys())]
            df["Name"] = names

            dfDict[csvPath] = df

        elif os.path.basename(csvPath) == "AnnotationManuelle2017.csv":
            df = pd.read_csv(csvPath,dtype=str,encoding = "ISO-8859-1",sep=";")

            idColName = "Nom"
            def preprocName(x):
                x = x.split("-")[0]
                if x.find("SLIDE") != -1:
                    x = x[:x.find("SLIDE")]
                return x
            def preprocWellInd(x):
                return x.split("-")[1]

            names = df[idColName].apply(preprocName)
            wellInds = df[idColName].apply(preprocWellInd)
            df = df[list(set(labelDict.keys()).intersection(set(df.columns)))]
            df["Name"] = names
            df["Well"] = wellInds

            dfDict[csvPath] = df

        elif os.path.basename(csvPath) == "export_emmanuelle.csv":

            df = pd.read_csv(csvPath,dtype=str,encoding = "ISO-8859-1",sep=",")
            idColName = "Patient Name"

            names = df[idColName].apply(preproc)
            df = df[["Well"]+list(labelDict.keys())]
            df["Name"] = names

            dfDict[csvPath] = df

        else:
            raise ValueError("Unkown ground truth csv file : ",csvPath)

    if not os.path.exists("../data/{}/timeImg/".format(dataset)):
        #Extracting and clustering the digits
        digitExtractor.clusterDigits(dataset,img_for_crop_nb)

    digExt = digitExtractor.DigitIdentifier(dataset)

    videoPaths = sorted(glob.glob("../data/{}/*.avi".format(dataset)))

    #Removing the videos with big problems
    videosToRemove = digitExtractor.getVideosToRemove()
    for vidName in videosToRemove:
        videoPaths.remove("../data/{}/".format(dataset)+vidName)

    noAnnot = 'video_name\n'
    multipleAnnot = 'video_name,annot1,annot2,annot3\n'

    for i,vidPath in enumerate(videoPaths):

        vidName = os.path.splitext(os.path.basename(vidPath))[0]
        patientName = vidName.split("-")[0]

        if not os.path.exists("../data/{}/annotations/{}_phases.csv".format(dataset,vidName)):

            phaseDict = {}
            imgCount = 0
            startOfCurrentPhase = 0
            currentLabel = None

            #Reading the video
            cap = cv2.VideoCapture(vidPath)
            ret, frame = cap.read()
            resDict = digExt.findDigits(frame,newVid=True)
            #print(resDict["time"])
            wellInd = resDict["wellInd"]

            rowList = []

            matchingCSVPaths = []
            for csvPath in dfDict.keys():

                rowLocBoolArray = (dfDict[csvPath]["Name"] == patientName)
                if rowLocBoolArray.sum() > 0:

                    patientLines = dfDict[csvPath].loc[rowLocBoolArray]

                    row = patientLines.loc[patientLines['Well'] == str(wellInd)]
                    colNames = []
                    for colName in ["Name","Well"]+list(labelDict.keys()):
                        if colName in list(row.columns):
                            colNames.append(colName)

                    rowList.append(row[colNames])
                    matchingCSVPaths.append(csvPath)

            if len(rowList) == 0:
                print(i,"/",len(videoPaths),": No annotation found")
                noAnnot += vidName + "\n"
            else:
                print(i,"/",len(videoPaths),vidName,":",len(rowList),"annotations found")

                if len(rowList) > 1:
                    line = vidName+','
                    for j in range(len(rowList)):
                        line += os.path.basename(matchingCSVPaths[j])+","

                    line += "\n"
                    multipleAnnot += line

                rows = pd.concat(rowList)

                #Using the first annotation found
                row = rows.iloc[0].to_frame().transpose()

                '''
                while ret:

                    #Select only the label columns of the desired well
                    row = row[list(labelDict.keys())]
                    for col in list(row.columns):
                        row[col] = row[col].apply(lambda x:x.replace(",",".") if type(x) == str else x).astype(float)

                    #Removes label columns that do not appear in the video (i.e. those with NaN value)
                    row = row.transpose()
                    row = row[np.isnan(row) == 0]
                    row = row.transpose()

                    #Getting the true label of the image
                    label = row.columns[max((resDict["time"] > row).sum(axis=1).item()-1,0)]

                    #Initialise currentLabel with the first label
                    if currentLabel is None:
                        currentLabel = label

                    #If this condition is true, the current frame belongs to a new phase
                    if label != currentLabel:
                        #Adding the start and end frames of last phase in the dict
                        phaseDict[currentLabel] = (startOfCurrentPhase,imgCount-1)
                        startOfCurrentPhase = imgCount
                        currentLabel = label

                    ret, frame = cap.read()
                    if not frame is None:
                        resDict = digExt.findDigits(frame)

                    imgCount +=1

                #Adding the last phase
                phaseDict[currentLabel] = (startOfCurrentPhase,imgCount-1)

                #Writing the start and end frames of each phase in a csv file
                with open("../data/{}/annotations/{}_phases.csv".format(dataset,vidName),"w") as text_file:
                    for label in labelDict.keys():
                        if label in phaseDict.keys():
                            print(label+","+str(phaseDict[label][0])+","+str(phaseDict[label][1]),file=text_file)
                '''

    with open("../data/{}/tooManyAnnot.csv".format(dataset),"w") as text_file:
        print(multipleAnnot,file=text_file)
    with open("../data/{}/noAnnot.csv".format(dataset),"w") as text_file:
        print(noAnnot,file=text_file)

def getLabels():
    return labelDict

def getReversedLabels():

    revDict = {}
    for key in labelDict.keys():
        revDict[labelDict[key]] = key

    return revDict

def main(argv=None):

    #Getting arguments from config file and command row
    #Building the arg reader
    argreader = ArgReader(argv)

    ########### PLOT SCORE EVOLUTION ALONG VIDEO ##################
    argreader.parser.add_argument('--dataset',type=str,metavar="DATASET",help='The dataset to format')
    argreader.parser.add_argument('--path_to_zip',type=str,metavar="PATH",help='The path to the zip file containing the dataset "small". Only used if it is the dataset "small" that is \
                                     being formated.',default="../data/embryo.zip")
    argreader.parser.add_argument('--path_to_folder',type=str,metavar="PATH",help='The path to the folder containing the dataset "big". Only used if it is the dataset "big" that is \
                                     being formated.',default="")

    argreader.parser.add_argument('--img_for_crop_nb',type=int,metavar="NB",help='The number of images from which to extract digits',default=2000)

    #Reading the comand row arg
    argreader.getRemainingArgs()

    #Getting the args from command row and config file
    args = argreader.args

    pd.options.display.width = 0

    if args.dataset == "small":
        formatDataSmall(args.dataset,args.path_to_zip,args.img_for_crop_nb)
    elif args.dataset == "big":
        formatDataBig(args.dataset,args.path_to_folder,args.img_for_crop_nb)

if __name__ == "__main__":
    main()
