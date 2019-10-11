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

    #Convert the xls files into csv files if it is not already done
    if len(glob.glob("../data/*.csv")) < len(glob.glob("../data/*.xls")):
        subprocess.call("libreoffice --headless --convert-to csv --outdir ../data/ ../data/*.xls",shell=True)

    digExt = digitExtractor.DigitIdentifier()

    for vidPath in vidPaths:

        csvPath = os.path.dirname(vidPath) + "/"+ os.path.basename(vidPath).split(" ")[0] + " EXPORT.csv"

        #Some xls space do not have the space between the id and the "EXPORT" string
        if not os.path.exists(csvPath):
            csvPath = os.path.dirname(vidPath) + "/"+ os.path.basename(vidPath).split(" ")[0] + "EXPORT.csv"

        if not os.path.exists(csvPath):
            raise OSError("Missing csv path for {}".format(vidPath))

        df = pd.read_csv(csvPath)[["Well"]+labels]

        #Reading the video
        videoImgCount = 0
        cap = cv2.VideoCapture(vidPath)
        ret, frame = cap.read()
        while ret:
            resDict = digExt.findDigits(frame)

            line = df.loc[df['Well'] == resDict["wellInd"]][labels]

            ret, frame = cap.read()

if __name__ == "__main__":
    formatData()
