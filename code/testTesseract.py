import pims
import PIL
import pytesseract
import scipy
from scipy import ndimage
import numpy
from skimage import feature
import numpy as np

def processFrame(img,dataset,totalCountDict):
	imgBin = img>128

	rightleft = imgBin.sum(axis=1)>2
	topbottom = imgBin.sum(axis=0)>2

	x,y = rightleft.argmax(),topbottom.argmax()
	width,heigth = rightleft.sum(),topbottom.sum()

	img = img[x:width+x,y:y+heigth]
	projX = (img<128).sum(axis=0)

	offSet = 0
	endReached = False
	digitList = []
	while not endReached:

		start = (projX[offSet:] != 0).argmax()+offSet
		length = (projX[start:] != 0).argmin()
		end = start+length

		if start==end:
			endReached=True
		else:
			digitList.append(img[:,start:end])
			offSet = end

	#Removing the character 'h' and the comma
	digitList.pop(-1)
	digitList.pop(-2)

	for i,dig in enumerate(digitList):
		PIL.Image.fromarray(dig).convert("RGB").save("../data/{}/timeImg/{}/digit{}.png".format(len(digitList)-i+1,totalCountDict["digit"+str(len(digitList)-i+1)]))
		totalCountDict["digit"+str(len(digitList)-i+1)] += 1


vid = pims.Video("../data/small/GC658_EMB3_DISCARD.avi")

for i in range(300):
    img = (vid[i][-50:,-50:].mean(axis=-1)).astype("uint8")

    #img[:,:12] = 255
    #img[:,-9:] = 255
    #img[:25,:] = 255
    #img[-10:,:] = 255
    #img_filt = scipy.ndimage.gaussian_filter(img,(3,3))

    PIL.Image.fromarray(img).convert("RGB").save("../data/small/notcropped{}.png".format(i))

    imgBin = img>128

    rightleft = imgBin.sum(axis=1)>2
    topbottom = imgBin.sum(axis=0)>2

    x,y = rightleft.argmax(),topbottom.argmax()
    width,heigth = rightleft.sum(),topbottom.sum()

    img = img[x:width+x,y:y+heigth]
    projX = (img<128).sum(axis=0)

    offSet = 0
    endReached = False
    digitList = []
    while not endReached:

        start = (projX[offSet:] != 0).argmax()+offSet
        length = (projX[start:] != 0).argmin()
        end = start+length

        if start==end:
            endReached=True
        else:
            digitList.append(img[:,start:end])
            offSet = end

    #Removing the character 'h' and the comma
    digitList.pop(-1)
    digitList.pop(-2)

    for dig in digitList:
        PIL.Image.fromarray(dig).convert("RGB").save("../data/small/digit{}_{}.png".format(i,offSet))

    #rightleft = np.repeat(rightleft[np.newaxis,:],img.shape[1],axis=0)
    #topbottom = np.repeat(topbottom[:,np.newaxis],img.shape[0],axis=1)

    #img[((rightleft<50)+(topbottom<100))>0] = 128

    img = PIL.Image.fromarray(img)
    #img = img.resize((400,400),resample=PIL.Image.BICUBIC)
    '''
    img = np.array(img).astype('int32')
    dx = ndimage.sobel(img, 0)  # horizontal derivative
    dy = ndimage.sobel(img, 1)  # vertical derivative
    mag = numpy.hypot(dx, dy)  # magnitude
    #mag = dx.astype("float64")
    mag *= 255.0 / numpy.max(mag)  # normalize (Q&D)
    mag= mag.astype("uint8")
    img = mag
    '''

    '''
    mask = np.zeros((6,6))
    mask[0:2,2:4] = 1
    mask[2:4,0:2] = 1
    mask[2:4,4:6] = 1
    mask[4:6,2:4] = 1

    mask = mask/mask.sum()

    res = scipy.signal.convolve2d(img,mask,mode="same").astype("uint8")

    img[img<50] = 0
    res[res<30] = 0
    res[res>100] = 255

    img = (img.astype("int")+res.astype("int"))
    img[img>255] = 255
    img = img.astype("uint8")

    PIL.Image.fromarray(res).convert("RGB").save("../data/small/cropped{}_conv.png".format(i))
    '''


    #img = PIL.Image.fromarray(img)

    #img = PIL.Image.fromarray(feature.canny(np.array(img), sigma=0.00000))

    img.convert("RGB").save("../data/small/cropped{}.png".format(i))
    res = pytesseract.image_to_data(img,output_type="dict")
    print(res["text"])
