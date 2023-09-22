import os

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

    assert x in [annot.value for annot in annot_enum],f"{x}, {task}, {annot_enum}"

    return list(annot_enum).index(annot_enum(x))

def make_annot_dict(dataset_name,dataset_path,is_train):

    datasets_names = set(dataset.value for dataset in Datasets)
    assert dataset_name in datasets_names,f"choose one dataset among {Datasets}"

    if Datasets(dataset_name) == Datasets.multi_center:
        make_annot_dic_func = make_multicenter_annot_dict
    else:
        make_annot_dic_func = make_dl4ivf_annot_dict

    annot_dict = make_annot_dic_func(dataset_path,is_train)

    return annot_dict

def make_dl4ivf_annot_dict(dataset_path,is_train,json_file_name="splits.json",annot_file_name="aggregated_annotations.csv"):

    json_file_path = os.path.join(dataset_path,json_file_name)
    with open(json_file_path, 'r') as fp:
        splits = json.load(fp)

    if is_train:
        split = splits["train"]
    else:
        split = splits["eval"]

    annot_file_path = os.path.join(dataset_path,annot_file_name)
    annot_csv = pd.read_csv(annot_file_path)
    annot_dict = {}
    
    def fill_dict(x):
        if x["image_name"] in split:
            annot_dict[x["image_name"]] = {task.value:preproc_annot_dl4ivf(x[task.value],task) for task in Tasks}

    annot_csv.apply(fill_dict,axis=1)
    
    return annot_dict

def make_multicenter_annot_dict(dataset_path,is_train):

    if is_train:
        annot_filename = "Gardner_train_silver.csv"
    else:
        annot_filename = "Gardner_test_gold_onlyGardnerScores.csv"

    annot_path = os.path.join(dataset_path,annot_filename)
    annot_csv = np.genfromtxt(annot_path,delimiter=";",dtype=str)

    dic = {}

    for row in annot_csv[1:]:
        sub_dic = {"exp":preproc_annot_multicenter(row[1]),
                   "icm":preproc_annot_multicenter(row[2]),
                   "te":preproc_annot_multicenter(row[3])}
        
        #Verify dic 
        annot_nb = 0 
        for key in sub_dic:
            if sub_dic[key] != NO_ANNOT:
                annot_nb += 1 
        
        assert annot_nb>0,f"Image {row[0]} from dataset {annot_filename} has no annotation: found {annot_nb} annotation for {len(sub_dic.keys())} keys."

        dic[row[0]] = sub_dic

    return dic 

def get_transform(size,phase='train'):

    if type(size) == int:
        kwargs={"size":(size,size)}
    else:
        kwargs={"size":(size[0], size[1])}

    if phase == 'train':
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

    def __init__(self,dataset_name,dataset_path,is_train,size):

        self.is_train = is_train
        self.annot_dict = make_annot_dict(dataset_name,dataset_path,is_train)

        self.image_list = sorted(self.annot_dict.keys())

        self.image_fold = os.path.join(dataset_path,"Images")
        self.transf = get_transform(size,phase='train' if is_train else "eval")

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

    for test in [True,False]:
        print("test is",test)

        for dataset in Datasets:
            print("\t",dataset.value)

            dataset_name = dataset.value
            dataset_path = path_dic[dataset]

            dataset = GradeDataset(dataset_name,dataset_path,False,(224,224))

            trainLoader = torch.utils.data.DataLoader(dataset=dataset, batch_size=2,
                                                    pin_memory=True, num_workers=0)

            images,annot = next(iter(trainLoader))

            print("\t\t",images.shape)

            torchvision.utils.save_image(images,f"../results/grade_dataset_test={test}.png")

            for key in annot:
                print("\t\t\t",key,annot[key])
