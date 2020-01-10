import pims
import PIL
import scipy
from scipy import ndimage
import numpy
from skimage import feature
import numpy as np
import cv2
import imageio
from skimage import img_as_ubyte
from skimage.transform import resize
vid = pims.Video("../data/small/GC658_EMB3_DISCARD.avi")
frameNb = 300

with imageio.get_writer("../data/GC658_EMB3_DISCARD_sobel.avi", mode='I',fps=30) as writer:

    for i in range(frameNb):

        img = (vid[i].mean(axis=-1)).astype("uint8")
        img[-50:] = 0

        img = np.array(img).astype('int32')
        dx = ndimage.sobel(img, 0)  # horizontal derivative
        dy = ndimage.sobel(img, 1)  # vertical derivative
        mag = numpy.hypot(dx, dy)  # magnitude
        #mag = dx.astype("float64")
        mag *= 255.0 / numpy.max(mag)  # normalize (Q&D)
        mag= mag.astype("uint8")

        sortedMagInd = mag.reshape(-1).argsort()[-10000:]
        sortedMag = np.sort(mag.reshape(-1))[-10000:]
        sortedMagInd = (sortedMagInd//mag.shape[0],sortedMagInd%mag.shape[0])
        sparseMag = np.zeros_like(mag)
        sparseMag[sortedMagInd[0],sortedMagInd[1]] = sortedMag
        mag = sparseMag
        img = mag

        nearestLowerDiv = img.shape[0]//16
        nearestHigherDiv = (nearestLowerDiv+1)*16
        img = resize(img, (nearestHigherDiv,nearestHigherDiv),anti_aliasing=True,mode="constant",order=0)*255

        print(img.mean())
        writer.append_data(img_as_ubyte(img.astype("uint8")))

vid = None
