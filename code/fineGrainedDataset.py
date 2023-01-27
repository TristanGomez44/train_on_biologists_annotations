""" CUB-200-2011 (Bird) Dataset
Created: Oct 11,2019 - Yuchong Gu
Revised: Oct 11,2019 - Yuchong Gu
"""
from ast import Lambda
import os
import pdb
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
                        cropRatio,brightness,saturation,add_patches=False,patch_res=14):

        self.image_path = {}
        self.image_label = {}
        self.root = "../data/"+root
        self.phase = phase
        self.resize = resize
        self.image_id = []
        self.num_classes = 200

        classes = [d.name for d in os.scandir(self.root) if d.is_dir()]

        if root.find("emb") != -1:
            classes.sort(key = lambda x:formatData.labelDict[x])
        else:
            classes.sort()

        class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}

        instances = []
        id = 0
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

                        id += 1

        # transform
        self.transform = get_transform(self.resize, self.phase,colorDataset=self.root.find("emb") == -1,\
                                        sqResizing=sqResizing,cropRatio=cropRatio,brightness=brightness,\
                                        saturation=saturation,add_patches=add_patches,patch_res=patch_res)

    def __getitem__(self, item):
        # get image id
        image_id = self.image_id[item]
        image = Image.open(self.image_path[image_id]).convert('RGB')  # (C, H, W)

        # image
        image = self.transform(image)

        if self.root.find("embryo_img_test") != -1:
            return image, self.image_label[image_id],self.image_path[image_id]
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

def get_transform(resize, phase='train',colorDataset=True,sqResizing=True,\
                    cropRatio=0.875,brightness=0.126,saturation=0.5,add_patches=False,patch_res=14):

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

    if phase == "train" and add_patches:
        def add_patches(img):

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
                print(values)
                mask = (mask > values[-1]) * 1.0
                mask = torch.nn.functional.interpolate(mask,img.shape[1:],mode="nearest")[0]

            return mask

        transf.append(transforms.Lambda(add_patches))

    transf = transforms.Compose(transf)

    return transf
