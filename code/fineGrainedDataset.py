""" CUB-200-2011 (Bird) Dataset
Created: Oct 11,2019 - Yuchong Gu
Revised: Oct 11,2019 - Yuchong Gu
"""
from ast import Lambda
import os
import pdb
from unittest.mock import patch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import numpy as np
import torch
import formatData

IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.tiff', '.webp')

class FineGrainedDataset(Dataset):
    """
    # Description:
        Dataset for retrieving CUB-200-2011 images and labels
    # Member Functions:
        __init__(self, phase, resize):  initializes a dataset
            phase:                      a string in ['train', 'val', 'test']
            resize:                     output shape/size of an image
        __getitem__(self, item):        returns an image
            item:                       the idex of image in the whole dataset
        __len__(self):                  returns the length of dataset
    """

    def __init__(self, root, phase,resize,sqResizing,\
                        cropRatio,brightness,saturation,other_image_batch=False):

        self.image_path = {}
        self.image_label = {}
        self.root = "../data/"+root
        self.phase = phase
        self.resize = resize
        self.image_id = []
        self.other_image_batch = other_image_batch
        classes = [d.name for d in os.scandir(self.root) if d.is_dir()]

        self.num_classes = len(classes)

        if root.find("emb") != -1:
            classes.sort(key = lambda x:formatData.labelDict[x])
        else:
            classes.sort()

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        instances = []
        id = 0
        self.class_to_id = {class_ind:[] for class_ind in np.arange(self.num_classes)}
        for target_class in sorted(class_to_idx.keys()):
            class_index = class_to_idx[target_class]
            target_dir = os.path.join(self.root, target_class)
            if not os.path.isdir(target_dir):
                continue
            for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
                for fname in sorted(fnames):
                    path = os.path.join(root, fname)
                    if is_valid_file(path):
                        item = path, class_index
                        instances.append(item)

                        self.image_path[id] = path
                        self.image_label[id] = class_index
                        self.image_id.append(id)

                        self.class_to_id[class_index].append(id)

                        id += 1

        # transform
        self.transform = get_transform(self.resize, self.phase,colorDataset=self.root.find("emb") == -1,\
                                        sqResizing=sqResizing,cropRatio=cropRatio,brightness=brightness,\
                                        saturation=saturation)

    def __getitem__(self, item):
        # get image id
        image_id = self.image_id[item]
        image = Image.open(self.image_path[image_id]).convert('RGB')  # (C, H, W)

        # image
        image = self.transform(image)

        if self.other_image_batch:

            candidate_class_labels = [i for i in range(self.num_classes)]
            candidate_class_labels.remove(self.image_label[image_id])
            
            rand_ind = np.random.randint(0,self.num_classes-1,size=(1,))[0]
            class_to_sample_from = candidate_class_labels[rand_ind]

            candidate_img_list = self.class_to_id[class_to_sample_from]
            rand_ind = np.random.randint(0,len(candidate_img_list),size=(1,))[0]
            img_to_sample_id = candidate_img_list[rand_ind]
            
            other_image = Image.open(self.image_path[img_to_sample_id]).convert('RGB')  # (C, H, W)
            other_image = self.transform(other_image)

            assert self.image_label[image_id] != class_to_sample_from

            return image,self.image_label[image_id],other_image 
        
        else:

            return image, self.image_label[image_id]

    def __len__(self):
        return len(self.image_id)

def is_valid_file(x):
    return has_file_allowed_extension(x, IMG_EXTENSIONS)

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    Args:
        filename (string): path to a file
        extensions (tuple of strings): extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    """
    return filename.lower().endswith(extensions)

def add_patches_func(img,patch_res):

    cent_x,cent_y = torch.randint(0,patch_res,size=(2,))
    var_x,var_y = torch.rand(size=(2,))+1

    x = torch.arange(patch_res).unsqueeze(0)
    y = torch.arange(patch_res).unsqueeze(1)

    mask = torch.exp(-((x-cent_x)**2)/var_x-((y-cent_y)**2)/var_y)
    mask = (mask - mask.min())/(mask.max() - mask.min())
    mask = mask.unsqueeze(0).unsqueeze(0)

    if torch.rand(size=(1,)) > 0.5:
        mask = (mask > 0.5)*1.0
        mask = torch.nn.functional.interpolate(mask,img.shape[1:],mode="bicubic",align_corners=False).clamp(min=0, max=1)[0]
    else:
        k = torch.randint(0,patch_res*patch_res,size=(1,)).item()
        values,_ = torch.topk(mask.view(-1),k,0,sorted=True)
        mask = (mask > values[-1]) * 1.0
        mask = torch.nn.functional.interpolate(mask,img.shape[1:],mode="nearest")[0]

    return mask

def get_transform(resize, phase='train',colorDataset=True,sqResizing=True,\
                    cropRatio=0.875,brightness=0.126,saturation=0.5):

    if sqResizing:
        kwargs={"size":(int(resize[0] / cropRatio), int(resize[1] / cropRatio))}
    else:
        kwargs={"size":int(resize[0] / cropRatio)}

    if phase == 'train':
        transf = [transforms.Resize(**kwargs),
                    transforms.RandomCrop(resize),
                    transforms.RandomHorizontalFlip(0.5)]

        if colorDataset:
            transf.extend([transforms.ColorJitter(brightness=brightness, saturation=saturation)])

    else:
        transf = [transforms.Resize(**kwargs),transforms.CenterCrop(resize)]

    transf.extend([transforms.ToTensor()])

    if colorDataset:
        transf.extend([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    transf = transforms.Compose(transf)

    return transf
