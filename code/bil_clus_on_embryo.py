import glob
import numpy as np
import sys
import os
from args import ArgReader

import modelBuilder
import load_data
import formatData
import trainVal
import torchvision

from torchvision import transforms

import random
import torch

import torch.nn.functional as F

def main(argv=None):
    argreader = ArgReader(argv)
    argreader = modelBuilder.addArgs(argreader)
    argreader = load_data.addArgs(argreader)
    argreader = trainVal.addInitArgs(argreader)

    argreader.getRemainingArgs()
    args = argreader.args

    net = modelBuilder.netBuilder(args)
    params = torch.load(args.init_path, map_location="cpu" if not args.cuda else None)
    net.load_state_dict(params, strict=False)
    net.eval()

    resize = 448
    preprocFunc = transforms.Compose([
                    transforms.ToPILImage(),
                    transforms.Resize(size=int(resize / 0.875)),
                    transforms.CenterCrop(resize),
                    transforms.ToTensor()])

    allPaths = sorted(glob.glob("../data/big/*avi"))

    allPaths = formatData.removeVid(allPaths,formatData.getNoAnnotVideos())
    allPaths = formatData.removeVid(allPaths,formatData.getEmptyAnnotVideos())
    allPaths = formatData.removeVid(allPaths,formatData.getTooFewPhaseVideos(6))

    random.seed(0)
    random.shuffle(allPaths)

    mode = "train"
    if mode == "train":
        testPaths = allPaths[:len(allPaths)//2]
    else:
        testPaths = allPaths[len(allPaths)//2:]

    bs = 15
    for m,path in enumerate(testPaths):
        print(m,"/",len(testPaths))
        grid = None
        nbImg = len(torchvision.io.read_video_timestamps(path,pts_unit="sec")[0])

        startFr = 100
        splitSizes = [bs for _ in range((nbImg-startFr)//bs)]+[(nbImg-startFr)%bs]
        frameInds_list = torch.split(torch.arange(startFr,nbImg),splitSizes)

        #for k,inds in enumerate(frameInds_list):
        for k,inds in enumerate(frameInds_list[:1]):

            print("\t",k,"/",len(frameInds_list),inds.shape)

            batch = loadFrames(path,inds.min(),inds.max(),preprocFunc)
            torchvision.utils.save_image(batch,"../vis/batchTest.png")

            if args.cuda:
                batch = batch.cuda()

            with torch.no_grad():
                retDict = net(batch)

            pred_012 = retDict["pred"].argmax(dim=-1)
            pred_01 = retDict["pred_01"].argmax(dim=-1)
            pred_0 = retDict["pred_0"].argmax(dim=-1)

            nbRequiredVec = torch.zeros_like(pred_012)
            for i in range(len(inds)):
                if pred_0[i] == pred_012[i]:
                    nbRequiredVec[i] = 1
                elif pred_01[i] == pred_012[i]:
                    nbRequiredVec[i] = 2
                else:
                    nbRequiredVec[i] = 3

                attMaps = retDict["attMaps"][i:i+1,:nbRequiredVec[i]]
                attMaps =(attMaps-attMaps.min())/(attMaps.max()-attMaps.min())
                padd = torch.zeros(attMaps.size(0),3-nbRequiredVec[i],attMaps.size(2),attMaps.size(3)).to(attMaps.device)
                attMaps = torch.cat((attMaps,padd),dim=1)
                attMaps = F.interpolate(attMaps, scale_factor=batch.size(2)*1.0/attMaps.size(2))
                attMaps = 0.8*attMaps+0.2*batch[i:i+1]

                img_attMaps = torch.cat((batch[i:i+1],attMaps),dim=0).cpu()

                if grid is None:
                    grid = img_attMaps
                else:
                    grid = torch.cat((grid,img_attMaps),dim=0)

        vidName = os.path.splitext(os.path.basename(path))[0]
        torchvision.utils.save_image(grid,"../vis/EMB8/{}_{}_{}.png".format(mode,args.model_id,vidName))


def loadFrames(videoPath,indStart,indEnd,preprocFunc):

    timeStamps = torchvision.io.read_video_timestamps(videoPath,pts_unit="sec")[0]
    startTime,endTime = timeStamps[indStart],timeStamps[indEnd]
    frameSeq = torchvision.io.read_video(videoPath,pts_unit="sec",start_pts=startTime,end_pts=endTime)[0].float()/255
    torchvision.utils.save_image(frameSeq.permute(0,3,1,2).float(),"../vis/batchTest_nopreproc.png")

	#Removing top border
    if frameSeq.size(1) > frameSeq.size(2):
        frameSeq = frameSeq[:,frameSeq.size(1)-frameSeq.size(2):]

    #Removing time
    frameSeq[:,-30:] = 0

    frameSeq = frameSeq.permute(0,3,1,2)

    procFrameList = []
    for frame in frameSeq:
        procFrame = preprocFunc(frame)
        procFrameList.append(procFrame.unsqueeze(0))
    procFrameBatch = torch.cat(procFrameList,dim=0)

    return procFrameBatch

def findVideos(propStart,propEnd,propSetIntFormat=False,shuffleData=False):

    #By setting dataset to "small+big", one can combine the two datasets

    allVideoPaths = []
    allVideoPaths += sorted(glob.glob("../data/qual/*.avi"))

    allVideoPaths = removeVid(allVideoPaths,formatData.getVideosToRemove())
    allVideoPaths = removeVid(allVideoPaths,formatData.getEmptyAnnotVideos())
    allVideoPaths = removeVid(allVideoPaths,formatData.getNoPhaseAnnotVideos())
    allVideoPaths = removeVid(allVideoPaths,formatData.getNoAnnotVideos())
    allVideoPaths = removeVid(allVideoPaths,formatData.getDoubles())
    allVideoPaths = removeVid(allVideoPaths,formatData.getNoEmbryo())

    if shuffleData:
        st = np.random.get_state()
        np.random.seed(1)
        np.random.shuffle(allVideoPaths)
        np.random.set_state(st)

    if propSetIntFormat:
        propStart /= 100
        propEnd /= 100

    if propStart < propEnd:
        videoPaths = np.array(allVideoPaths)[int(propStart*len(allVideoPaths)):int(propEnd*len(allVideoPaths))]
    else:
        videoPaths = allVideoPaths[int(propStart*len(allVideoPaths)):]
        videoPaths += allVideoPaths[:int(propEnd*len(allVideoPaths))]
        videoPaths = np.array(videoPaths)

    return videoPaths



if __name__ == "__main__":
    main()
