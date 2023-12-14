import os
import math 
import json 
from collections import defaultdict

import torch
from PIL import Image
import numpy as np
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import Dataset

from enums import annot_enum_dic,Datasets,Tasks

NO_ANNOT = -1

def preproc_annot_multicenter(x):
    if x.isdigit():
        return int(x)
    else:
        return NO_ANNOT

def preproc_annot_dl4ivf(x,task):
    if type(x) is str:
        annot_enum = annot_enum_dic[task]
        return list(annot_enum).index(annot_enum(x))
    else:
        return x

def make_annot_dict(dataset_name,dataset_path,mode,distr_learn,zmos):

    datasets_names = set(dataset.value for dataset in Datasets)
    assert dataset_name in datasets_names,f"choose one dataset among {Datasets}"

    if Datasets(dataset_name) == Datasets.multi_center:
        make_annot_dic_func = make_multicenter_annot_dict
    else:
        make_annot_dic_func = make_dl4ivf_annot_dict

    annot_dict = make_annot_dic_func(dataset_path,mode,distr_learn=distr_learn,zmos=zmos)

    return annot_dict

def get_videos_in_split(dataset_path,mode,split_file_name):
    json_file_path = os.path.join(dataset_path,split_file_name)
    with open(json_file_path, 'r') as fp:
        splits = json.load(fp)
    return splits[mode]   

def make_dl4ivf_annot_dict(dataset_path,mode,split_file_name="splits.json",annot_file_name="aggregated_annotations.csv",distr_learn=False,zmos=False):

    split = get_videos_in_split(dataset_path,mode,split_file_name)

    if zmos:
        annot_file_name = annot_file_name.replace(".csv","_ZRECMOS.csv")

    annot_file_path = os.path.join(dataset_path,annot_file_name)
    annot_csv = pd.read_csv(annot_file_path,delimiter=" ")

    if not distr_learn:
        annot_dict = {}
        def fill_dict(x):
            if x["image_name"] in split:
                annot_dict[x["image_name"]] = {task.value:preproc_annot_dl4ivf(x[task.value],task) for task in Tasks}

        annot_csv.apply(fill_dict,axis=1)
    else:
        annot_dict = defaultdict(lambda:defaultdict(lambda:{}))

        for task in Tasks:

            possible_values = list(annot_enum_dic[task])
            def fill_dict(x):
                if x["image_name"] in split:
                    values = torch.tensor([x[value.value] for value in possible_values]).int()
                    annot_dict[x["image_name"]][task.value] = values/values.sum()

            distr_annot_csv = pd.read_csv(annot_file_path.replace(".csv","_"+task.value+".csv"),delimiter=" ")

            distr_annot_csv.apply(fill_dict,axis=1)
            
    return annot_dict

def make_multicenter_annot_dict(dataset_path,mode,split_file_name="splits.json",train_annot_file_name="Gardner_train_silver.csv",eval_annot_file_name="Gardner_test_gold_onlyGardnerScores.csv",distr_learn=None,zmos=None):

    assert distr_learn is None,"Cannot use distribution learning on this dataset."
    assert zmos is None,"Cannot use ZMOS scores on this dataset."

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

def get_transform(size,mode='train',random_transf=True):

    if type(size) == int:
        kwargs={"size":(size,size)}
    else:
        kwargs={"size":(size[0], size[1])}

    if mode == 'train' and random_transf:
        transf = [transforms.RandomResizedCrop(size,scale=(0.9,1),ratio=(1,1)),
                    transforms.RandomVerticalFlip(0.5),
                    transforms.RandomHorizontalFlip(0.5)]#,
                    #transforms.RandomRotation(degrees=180)]
    else:
        transf = [transforms.Resize(**kwargs)]

    transf.append(transforms.ToTensor())
    transf.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
    transf = transforms.Compose(transf)

    return transf


class GradeDataset(Dataset):

    def __init__(self,dataset_name,dataset_path,mode,size,random_transf=True,distr_learn=False,zmos=False):

        self.mode = mode
        self.annot_dict = make_annot_dict(dataset_name,dataset_path,mode,distr_learn,zmos)

        self.image_list = sorted(self.annot_dict.keys())

        self.image_fold = os.path.join(dataset_path,"Images")
        self.transf = get_transform(size,mode=mode,random_transf=random_transf)

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