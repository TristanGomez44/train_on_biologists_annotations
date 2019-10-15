import pims
import numpy as np
import xml.etree.ElementTree as ET
import os
import processResults
import subprocess

def getVideoFrameNb(videoPath):
    if hasattr(pims.Video(videoPath),"_len"):
        frameNb = pims.Video(videoPath)._len
    else:
        fps = getVideoFPS(videoPath)
        frameNb = round(float(pims.Video(videoPath)._duration)*fps)

    return frameNb

def getVideoFPS(videoPath):
    ''' Get the number of frame per sencond of a video.'''

    pimsVid = pims.Video(videoPath)

    if hasattr(pimsVid,"_frame_rate"):
        return float(pims.Video(videoPath)._frame_rate)
    else:
        res = subprocess.check_output("ffprobe -v 0 -of csv=p=0 -select_streams v:0 -show_entries stream=r_frame_rate {}".format(videoPath),shell=True)
        res = str(res)[:str(res).find("\\n")].replace("'","").replace("b","").split("/")
        fps = int(res[0])/int(res[1])
        return fps

def findNumbers(x):
    '''Extracts the numbers of a string and returns them as an integer'''

    return int((''.join(xi for xi in str(x) if xi.isdigit())))

def findLastNumbers(weightFileName):
    '''Extract the epoch number of a weith file name.

    Extract the epoch number in a weight file which name will be like : "clustDetectNet2_epoch45".
    If this string if fed in this function, it will return the integer 45.

    Args:
        weightFileName (string): the weight file name
    Returns: the epoch number

    '''

    i=0
    res = ""
    allSeqFound = False
    while i<len(weightFileName) and not allSeqFound:
        if not weightFileName[len(weightFileName)-i-1].isdigit():
            allSeqFound = True
        else:
            res += weightFileName[len(weightFileName)-i-1]
        i+=1

    res = res[::-1]

    return int(res)
