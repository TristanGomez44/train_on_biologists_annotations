import os
import math 

from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from enums import annot_enum_dic,Datasets,Tasks
import json 

NO_ANNOT = -1

def preproc_annot_multicenter(x):
    if x.isdigit():
        return int(x)
    else:
        return NO_ANNOT

def preproc_annot_dl4ivf(x,task):

    annot_enum = annot_enum_dic[task]

    if type(x) is str or not math.isnan(x):
        assert x in [annot.value for annot in annot_enum],f"{x}, {task}, {annot_enum}"
    
        return list(annot_enum).index(annot_enum(x))

    else:
        return NO_ANNOT

def make_annot_dict(dataset_name,dataset_path,mode):

    datasets_names = set(dataset.value for dataset in Datasets)
    assert dataset_name in datasets_names,f"choose one dataset among {Datasets}"

    if Datasets(dataset_name) == Datasets.multi_center:
        make_annot_dic_func = make_multicenter_annot_dict
    else:
        make_annot_dic_func = make_dl4ivf_annot_dict

    annot_dict = make_annot_dic_func(dataset_path,mode)

    return annot_dict

def get_videos_in_split(dataset_path,mode,split_file_name):
    json_file_path = os.path.join(dataset_path,split_file_name)
    with open(json_file_path, 'r') as fp:
        splits = json.load(fp)
    return splits[mode]   

def make_dl4ivf_annot_dict(dataset_path,mode,split_file_name="splits.json",annot_file_name="aggregated_annotations.csv"):

    split = get_videos_in_split(dataset_path,mode,split_file_name)

    annot_file_path = os.path.join(dataset_path,annot_file_name)
    annot_csv = pd.read_csv(annot_file_path,delimiter=" ")
    annot_dict = {}
 
    def fill_dict(x):
        if x["image_name"] in split:
            annot_dict[x["image_name"]] = {task.value:preproc_annot_dl4ivf(x[task.value],task) for task in Tasks}

    annot_csv.apply(fill_dict,axis=1)
    return annot_dict

def make_multicenter_annot_dict(dataset_path,mode,split_file_name="splits.json",train_annot_file_name="Gardner_train_silver.csv",eval_annot_file_name="Gardner_test_gold_onlyGardnerScores.csv"):

    split = get_videos_in_split(dataset_path,mode,split_file_name)

    if mode in ["train","val"]:
        annot_filename = train_annot_file_name
    else:
        annot_filename = eval_annot_file_name

    annot_path = os.path.join(dataset_path,annot_filename)
    annot_csv = np.genfromtxt(annot_path,delimiter=";",dtype=str)

    dic = {}

    for row in annot_csv[1:]:

        img_name = row[0]

        if img_name in split:

            sub_dic = {"EXP":preproc_annot_multicenter(row[1]),
                    "ICM":preproc_annot_multicenter(row[2]),
                    "TE":preproc_annot_multicenter(row[3])}
            
            #Verify dic 
            annot_nb = 0 
            for key in sub_dic:
                if sub_dic[key] != NO_ANNOT:
                    annot_nb += 1 
            
            assert annot_nb>0,f"Image {img_name} from dataset {annot_filename} has no annotation: found {annot_nb} annotation for {len(sub_dic.keys())} keys."

            dic[img_name] = sub_dic

    return dic 

def get_transform(size,mode='train'):

    if type(size) == int:
        kwargs={"size":(size,size)}
    else:
        kwargs={"size":(size[0], size[1])}

    if mode == 'train':
        transf = [transforms.RandomResizedCrop(size,scale=(0.9,1),ratio=(1,1)),
                    transforms.RandomVerticalFlip(0.5),
                    transforms.RandomHorizontalFlip(0.5),
                    transforms.RandomRotation(degrees=180)]
    else:
        transf = [transforms.Resize(**kwargs)]

    transf.append(transforms.ToTensor())
    transf.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    transf = transforms.Compose(transf)

    return transf


class GradeDataset(Dataset):

    def __init__(self,dataset_name,dataset_path,mode,size):

        self.mode = mode
        self.annot_dict = make_annot_dict(dataset_name,dataset_path,mode)

        self.image_list = sorted(self.annot_dict.keys())

        self.image_fold = os.path.join(dataset_path,"Images")
        self.transf = get_transform(size,mode=mode)

    def __getitem__(self, item):

        image_name = self.image_list[item]
        image_path = os.path.join(self.image_fold,image_name)
        image = Image.open(image_path).convert('RGB')

        image = self.transf(image)

        annot = self.annot_dict[image_name]

        return image,annot 

    def __len__(self):
        return len(self.image_list)

if __name__ == "__main__":
    import torch
    import torchvision

    path_dic = {Datasets.multi_center:"../data/Blastocyst_Dataset",Datasets.dl4ivf:"../data/dl4ivf_blastocysts/"}

    for dataset in Datasets:
        print(dataset.value)
                
        imgs_list = []

        for mode in ["train","val","test"]:
            print("\t",mode)

            dataset_name = dataset.value
            dataset_path = path_dic[dataset]

            torch_dataset = GradeDataset(dataset_name,dataset_path,mode,(224,224))

            imgs_list.append(torch_dataset.image_list)

            trainLoader = torch.utils.data.DataLoader(dataset=torch_dataset, batch_size=2,
                                                    pin_memory=True, num_workers=0)

            images,annot = next(iter(trainLoader))

            print("\t\t",images.shape)

            torchvision.utils.save_image(images,f"../results/grade_dataset_{mode}.png")

            for key in annot:
                print("\t\t\t",key,annot[key])

        print("\ttrain-val intersec",set(imgs_list[0]).intersection(imgs_list[1]))
        print("\ttrain-test intersec",set(imgs_list[0]).intersection(imgs_list[2]))
        print("\tval-test intersec",set(imgs_list[1]).intersection(imgs_list[2]))