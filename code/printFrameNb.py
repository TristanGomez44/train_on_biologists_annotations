import numpy as np
import glob
import utils
res = ""

for vidPath in sorted(glob.glob("../data/big/*avi")):
    res += str(utils.getVideoFrameNb(vidPath)) + "\n"

with open("frameNb.csv","w") as text_file:
    print(res,file=text_file)
