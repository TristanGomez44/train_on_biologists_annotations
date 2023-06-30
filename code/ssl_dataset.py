import os,glob,sys

from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import numpy as np
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms
from torch.utils.data import Dataset
import torch 

class SSLDataset(Dataset):

    def __init__(self,dataset_path,mode,train_prop,val_prop,img_size,img_ext="jpeg",augment=True,only_center_plane=False) -> None:
        super().__init__()

        video_pattern = os.path.join(dataset_path,"D*/")
        self.video_paths = sorted(glob.glob(video_pattern))

        np.random.seed(0)
        np.random.shuffle(self.video_paths)
        
        vid_nb = len(self.video_paths)
        if mode == "train":
            start,end = 0,int(vid_nb*train_prop)
        elif mode == "val":
            start,end = int(vid_nb*train_prop),int(vid_nb*(train_prop+val_prop))
        else:
            start,end = int(vid_nb*(train_prop+val_prop)),vid_nb
        
        self.video_paths = self.video_paths[start:end]

        self.img_ext = img_ext
        self.transf = get_transform(img_size,augment=augment)
        self.only_center_plane = only_center_plane

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self,i):

        video_path = self.video_paths[i]

        if self.only_center_plane:
            focal_plane = video_path+"/F0"
        else:
            all_focal_plane_paths = sorted(glob.glob(video_path+"/F*"),key=get_focal_plane_value)
            focal_plane_ind = torch.randint(0,high=len(all_focal_plane_paths)-1,size=(1,)).item()
            focal_plane = all_focal_plane_paths[focal_plane_ind]

        all_img_paths = sorted(glob.glob(focal_plane+"/*."+self.img_ext),key=get_frame_ind)
        frame_ind = torch.randint(0,high=len(all_img_paths)-1,size=(1,)).item()

        if torch.rand((1,)).item() > 0.5 and not self.only_center_plane:
            #Focal plane neighbors
            img_list = []
            for ind in [focal_plane_ind,focal_plane_ind+1]:
                focal_plane = all_focal_plane_paths[ind]
                all_img_paths = sorted(glob.glob(focal_plane+"/*."+self.img_ext),key=get_frame_ind)
                
                assert len(all_img_paths) > 0,focal_plane
                assert frame_ind < len(all_img_paths),focal_plane+" "+str(frame_ind)+" "+str(len(all_img_paths))
            
                img_path = all_img_paths[frame_ind]
                img = Image.open(img_path)
                img = convert_to_rgb(img)
                img = self.transf(img)
                img_list.append(img)
        else:
            #Time neighbors
            img_list = []
            for ind in [frame_ind,frame_ind+1]:
                img_path = all_img_paths[ind]
                img = Image.open(img_path)
                img = convert_to_rgb(img)
                img = self.transf(img)
                img_list.append(img)

        return img_list[0],img_list[1]

def convert_to_rgb(img):
    img = np.array(img)
    img = img[:,:,np.newaxis]
    img = np.repeat(img,3,axis=2)
    img = Image.fromarray(img)
    return img 

def get_focal_plane_value(path):
    return int(path.split("/")[-1][1:])

def get_frame_ind(path):
    file_name = os.path.splitext(os.path.basename(path))[0]
    frame_ind = int(file_name.split("RUN")[1])
    return frame_ind

#Construct a basic torchvision transform
def get_transform(img_size,augment=True):

    if augment:
        transf = [transforms.RandomResizedCrop(img_size,scale=(0.9,1),ratio=(1,1)),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomRotation(degrees=180)]
    else:
        transf = [transforms.Resize(img_size)]

    transf.extend([transforms.ToTensor()])
    transf.extend([transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                        std=[0.229, 0.224, 0.225])])

    transf = transforms.Compose(transf)

    return transf

