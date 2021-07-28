
import glob
import pandas as pd
import numpy as np 
import sys 
import os 
import formatData,digitExtractor
def removeSuff(x):
    if len(x.split("_")) == 4:
        x = "_".join(x.split("_")[:-1])
    return x

def fromBig(x):

    if x.find("-") != -1:

        if len(x.split("-")) > 2:
            patID = "-".join(x.split("-")[:-1])
            embNb = x.split("-")[-1]
        else:
            patID,embNb = x.split("-")
    else:
        x_split = x.split("_")

        if len(x_split) == 3:
            patID,embNb,_ = x_split
            embNb = embNb.replace("EMB","")
        else:
            try:
                patID = x_split[0]
                embNb = x_split[2]
            except IndexError:
                print(x,x_split)
                sys.exit(0)

    if len(embNb.split("_")) == 3:
        embNb = embNb.split("_")[0]
    elif len(embNb.split("_")) != 1:
        raise ValueError("Unkown case",x,patID,embNb)

    if patID.find("SLIDE") != -1:
        patID,slide = patID.split("SLIDE")
        slide = int(slide.split("-")[0])
    else:
        slide = -1 

    return patID,int(embNb),slide

def getSlideInd(x):

    origX = x

    if x.find("SLIDES SUITE CULTURE") != -1:
        x = -1
    elif x.find("1ERE SLIDE") != -1:
        x = 1
    elif x.find("2EME SLIDE") != -1:
        x = 2   
    elif x.find("TEST-SLIDE-VIDE") != -1:
        x = -1
    elif x.find("SLIDE") != -1:
        x = x.split("SLIDE")[1].split("/")[0]
        if x == "":
            x = -1
        elif len(x.split(" ")) > 2:
            x = x.split(" ")[1]
        else:
            x = x.replace(" ","") 
    else:
        x = -1

    if not pd.notnull(x):
        raise ValueError("NAN",x)

    try:
        int(x)
    except ValueError:
        print("erro",origX,"final",x)
        sys.exit(0)

    return int(x)

def formatPatID(x):

    if x.find("/") != -1:
        start = x.split("/")[0].replace(" ","")
        end = x.split("/")[1].split("SLIDE")[0].replace(" ","")

        x = start+end
        
    else:
        x = x.replace(" ","")

    return x.upper()

def findFrameInd(path):
    fileName = os.path.splitext(os.path.basename(path))[0]
    frameInd = int(fileName.split("RUN")[1])
    return frameInd

def apndRowToFold(dic,rows,fold):
    for row in rows:
        dic[row] = fold
    return dic

rootCeph ="/mnt/cephfs-ec/DL4IVF/"
rootHDD= "/media/E144069X/Seagate Expansion Drive/extractions/"

hdd = os.path.exists(rootHDD)

if hdd:
    root = rootHDD 
    rootDic = {"2017":"/media/E144069X/Seagate Expansion Drive/extractions/",\
                "2018":"/media/E144069X/Seagate Expansion Drive/extractions/",\
                "2019":"/media/E144069X/Seagate Expansion Drive/extractions/",\
                "2016":"/media/E144069X/DL4IVF/extractions/",\
                "2011_2015":"/media/E144069X/DL4IVF/extractions/"}
else:
    root = rootCeph 
    rootDic = {year:rootCeph for year in ["2017","2018","2019","2016","2011_2015"]}
    
years = list(rootDic.keys())

df = pd.read_csv("../data/EXPORT 2017-2019.csv",delimiter=";")
df["Slide ID"] = df["Slide ID"].apply(removeSuff)
df["Slide Ind"] = df["Patient Name"].apply(getSlideInd)
df["Patient Name"] = df["Patient Name"].apply(formatPatID)

df_old = pd.read_csv("../data/export 2011-2016.csv",delimiter=";")
df_old["Slide Ind"] = df_old["Patient Name"].apply(getSlideInd)
df_old["Patient Name"] = df_old["Patient Name"].apply(formatPatID)

if not os.path.exists("rowToFold.npy"):
    rowToFold = {}
    rowToFold_old = {}

    for year in years:

        print(year)

        if hdd:
            vidPaths = sorted(glob.glob(rootDic[year]+"/"+year+"/*/"))
            print(rootDic[year]+"/"+year+"/*/")
        else:
            vidPaths = sorted(glob.glob(rootDic[year]+"/D"+year+"*/"))
            print(rootDic[year]+"/"+year+"*/")

        for i,path in enumerate(vidPaths):
            vidName = path.split("/")[-2]

            if len(vidName.split("_")) == 4:  
                vidNameRoot = "_".join(vidName.split("_")[:-1])
                embNb = vidName.split("_")[-1].replace("D","").replace("P","").replace("-","")      
            else:
                vidNameRoot = vidName.split("-")[0]
                embNb = vidName.split("-")[-1]  

            if i % 200 == 0:
                print("\t",i,"/",len(vidPaths))

            slideIdBin = df["Slide ID"] == vidNameRoot
            slideIdBin_old = df_old["Slide ID"] == vidNameRoot
            
            if vidNameRoot in list(df["Slide ID"]):
                if embNb in list(df["Embryo ID"][slideIdBin]):
                    rowToFold = apndRowToFold(rowToFold,df.index[slideIdBin & (df["Embryo ID"] == embNb)].tolist(),vidName)
                elif int(embNb) in list(df["Well"][slideIdBin]):
                    rowToFold = apndRowToFold(rowToFold,df.index[slideIdBin & (df["Well"] == int(embNb))].tolist(),vidName)
                else:
                    print("Missing emb",vidName,vidNameRoot,embNb)
                    print(list(df["Embryo ID"][slideIdBin]))
                    print(list(df["Well"][slideIdBin]))

            elif vidNameRoot in list(df_old["Slide ID"]):
                if embNb in list(df_old["Embryo ID"][slideIdBin_old]):
                    rowToFold_old = apndRowToFold(rowToFold_old,df_old.index[slideIdBin_old & (df_old["Embryo ID"] == embNb)].tolist(),vidName)
                elif int(embNb) in list(df_old["Well"][slideIdBin_old]):
                    rowToFold_old = apndRowToFold(rowToFold_old,df_old.index[slideIdBin_old & (df_old["Well"] == int(embNb))].tolist(),vidName)
                else:
                    print("Missing emb",vidName,vidNameRoot,embNb)
                    print(list(df_old["Embryo ID"][slideIdBin_old]))
                    print(list(df_old["Well"][slideIdBin_old]))
            
            else:
                print("Missing vid",vidNameRoot)

    np.save("rowToFold.npy",rowToFold)
    np.save("rowToFold_old.npy",rowToFold_old)
else:
    rowToFold = np.load("rowToFold.npy",allow_pickle=True).item()
    rowToFold_old = np.load("rowToFold_old.npy",allow_pickle=True).item()

rows_big = []

#root_dataset = "../embryo_dataset/"
#vidPaths = sorted(glob.glob(root_dataset + "/*.avi"))

vidPaths = glob.glob("../data/big/*.avi")+glob.glob("../data/small/*.avi")
vidPaths = sorted(vidPaths)

def removeVid(videoPaths,videoToRemoveNames):
    #Removing videos with bad format
    vidsToRemove = []
    for vidPath in videoPaths:
        for vidName in videoToRemoveNames:
            if os.path.splitext(os.path.basename(vidPath))[0] == vidName:
                vidsToRemove.append(vidPath)
                print(vidPath,os.path.splitext(os.path.basename(vidPath))[0],vidName)
    for vidPath in vidsToRemove:
        videoPaths.remove(vidPath)

    return videoPaths

vidPaths = removeVid(vidPaths,digitExtractor.getVideosToRemove())
vidPaths = removeVid(vidPaths,formatData.getNoAnnotVideos())
vidPaths = removeVid(vidPaths,formatData.getEmptyAnnotVideos())
vidPaths = removeVid(vidPaths,formatData.getTooFewPhaseVideos(6))

missingVid = []
oldVid = []
newVid = []

if not os.path.exists("vidToRow.npy"):

    vidToRow = {}
    vidToRow_old = {}
    vidToRow_pat = {}

    notFoundNb = 0
    for i,vid in enumerate(vidPaths):

        if i % 200 == 0:
            print(i,"/",len(vidPaths))

        vidName = os.path.splitext(os.path.basename(vid))[0]

        patientID,embNb,slide = fromBig(vidName)

        found = False

        if slide > -1:
            slideValues = list(set(df["Slide Ind"][df["Patient Name"] == patientID].tolist()))
            

            if len(slideValues)> 0 and not pd.notnull(slideValues[0]):
                print("NEW")
                print(slideValues)
                print(patientID,embNb,slide)
   



            if len(slideValues) == 1 and int(slideValues[0]) == -1:
                patIdBin = (df["Patient Name"] == patientID)
            else:
                patIdBin = (df["Patient Name"] == patientID) & ((df["Slide Ind"] == slide) | (df["Slide Ind"] == str(slide)))
        else:
            patIdBin = (df["Patient Name"] == patientID)

        if patientID in list(df["Patient Name"]):
            if embNb in list(df["Embryo ID"][patIdBin]):
                rows_big.extend(df.index[(patIdBin) & (df["Embryo ID"] == embNb)].tolist()) 
                newVid.append(vidName)
                vidToRow[vidName] = df.index[(patIdBin) & (df["Embryo ID"] == embNb)].tolist()
                found = True
            elif int(embNb) in list(df["Well"][patIdBin]):
                rows_big.extend(df.index[(patIdBin) & (df["Well"] == int(embNb))].tolist()) 
                newVid.append(vidName)
                vidToRow[vidName] = df.index[(patIdBin) & (df["Well"] == int(embNb))].tolist()
                found = True
            else:
                print(patientID,embNb,list(df["Embryo ID"][patIdBin]),list(df["Well"][patIdBin]))
                if len(list(df["Embryo ID"][patIdBin])) == 0:
                    print(patientID,slide,embNb)
                    print(df["Embryo ID"][patIdBin])

        if slide > -1:
            slideValues = list(set(df_old["Slide Ind"][df_old["Patient Name"] == patientID].tolist()))
            
            
            if len(slideValues)> 0 and not pd.notnull(slideValues[0]):
                print("OLD")
                print(slideValues)
                print(patientID,embNb,slide)
         


            if len(slideValues) == 1 and int(slideValues[0]) == -1:
                patIdBin = (df_old["Patient Name"] == patientID)
            else:
                patIdBin = (df_old["Patient Name"] == patientID) & ((df_old["Slide Ind"] == slide) | (df_old["Slide Ind"] == str(slide)))
        else:
            patIdBin = (df_old["Patient Name"] == patientID)           

        if patientID in list(df_old["Patient Name"]):
            if embNb in list(df_old["Embryo ID"][patIdBin]):
                rows_big.extend(df_old.index[(patIdBin) & (df_old["Embryo ID"] == embNb)].tolist())
                oldVid.append(vidName)
                vidToRow_old[vidName] = df_old.index[(patIdBin) & (df_old["Embryo ID"] == embNb)].tolist()
                found = True
            elif int(embNb) in list(df_old["Well"][patIdBin]):
                rows_big.extend(df_old.index[(patIdBin) & (df_old["Well"] == int(embNb))].tolist()) 
                oldVid.append(vidName)
                vidToRow_old[vidName] = df_old.index[(patIdBin) & (df_old["Well"] == int(embNb))].tolist()
                found = True
            else:
                print(patientID,embNb,list(df_old["Embryo ID"][patIdBin]),list(df_old["Well"][patIdBin]))
                if len(list(df_old["Embryo ID"][patIdBin])) == 0:
                    print(patientID,slide,embNb)
                    print(df_old["Embryo ID"][patIdBin])

        if os.path.exists("../data/small/{}_EXPORT.csv".format(patientID)):
            df_pat = pd.read_csv("../data/small/{}_EXPORT.csv".format(patientID),delimiter=",")

            if embNb in list(df_pat["Embryo ID"]):
                rows_big.extend(df_pat.index[df_pat["Embryo ID"] == embNb].tolist())
                oldVid.append(vidName)
                vidToRow_pat[vidName] = df_pat.index[df_pat["Embryo ID"] == embNb].tolist()
                found = True
            elif int(embNb) in list(df_pat["Well"]):
                rows_big.extend(df_pat.index[df_pat["Well"] == int(embNb)].tolist()) 
                oldVid.append(vidName)
                vidToRow_pat[vidName] = df_pat.index[df_pat["Well"] == int(embNb)].tolist()
                found = True
            #else:
            #    print(patientID,embNb,list(df_pat["Embryo ID"][patIdBin]),list(df_pat["Well"][patIdBin]))

        if not found:
            missingVid.append(vidName)

    np.save("vidToRow.npy",vidToRow)
    np.save("vidToRow_old.npy",vidToRow_old)
    np.save("vidToRow_pat.npy",vidToRow_pat)

    oldVid = list(set(oldVid))
    newVid = list(set(newVid))

    missingVid = sorted(list(set(missingVid) - set(oldVid) - set(newVid)))

    print("Missing vids : ",len(missingVid),"/",len(vidPaths))
    print("Old vids : ",len(oldVid),"/",len(vidPaths))
    print("New vids : ",len(newVid),"/",len(vidPaths))
 
    with open("../data/videoMissingInTheCSV.csv","w") as file:
        for elem in missingVid:
            print(elem,file=file)


else:
    vidToRow = np.load("vidToRow.npy",allow_pickle=True).item()
    vidToRow_old = np.load("vidToRow_old.npy",allow_pickle=True).item()

def checkEmbNb(vid,foldName):
    
    if len(vid.split("-")) == 1:
        vidSplit = vid.split("EMB")[1]
        if len(vidSplit.split("_")) > 2:
            vidEmbInd = int(vidSplit.split("_")[1])
        else:
            vidEmbInd = int(vidSplit.split("_")[0])              
    else:
        vidEmbInd = int(vid.split("-")[-1].split("_")[0])
    apiEmbInd = int(foldName.split("-")[-1])

    return vidEmbInd == apiEmbInd

missingList = []

match_nb = 0
tryNb = 0
true_match = 0
for vid in vidToRow:
    match_found =False
    for row in vidToRow[vid]:
        if row in rowToFold:
            match = checkEmbNb(vid,rowToFold[row])

            if match:
                match_found = True 
                break
    if not match_found:
        missingList.append(vid)
    tryNb += 1

for vid in vidToRow_old:
    match_found =False
    for row in vidToRow_old[vid]:
        if row in rowToFold_old:
            match = checkEmbNb(vid,rowToFold_old[row])        

            if match:
                match_found = True
                break
    if not match_found:
        missingList.append(vid)
    tryNb += 1

with open("../data/missingCorrVids.txt","w") as file:
    for i in range(len(missingList)):
        print(missingList[i],file=file)

print(len(missingList),len(vidToRow)+len(vidToRow_old))