import glob
import numpy as np
import os
import utils

paths = sorted(glob.glob("../data/*/annotations/*_phases.csv"))

for i,phasesFilePath in enumerate(paths):

    if i%50 == 0:
        print(i,"/",len(paths))

    if os.stat(phasesFilePath).st_size > 0:
        phases = np.genfromtxt(phasesFilePath,delimiter=",",dtype=str)

        videoName = "_".join(os.path.basename(phasesFilePath).split("_")[:-1])
        videoDir = "/".join(os.path.dirname(phasesFilePath).split("/")[:-1])
        frameNb = utils.getVideoFrameNb(os.path.join(videoDir,videoName+".avi"))

        if len(phases.shape) ==  2:

            if int(phases[-1,1])< frameNb:
                phases[-1,1] = str(int(phases[-1,1])+1)
                print("Fixing file",videoName)
            elif int(phases[-1,1]) > frameNb:
                raise ValueError("WTF")
        else:
            if int(phases[-1]) < frameNb:
                phases[-1] = str(int(phases[-1])+1)
                print("Fixing file",videoName)
            elif int(phases[-1]) > frameNb:
                raise ValueError("WTF")

        print(phases[-1])
        np.savetxt(phasesFilePath,phases,delimiter=",",fmt="%s")
