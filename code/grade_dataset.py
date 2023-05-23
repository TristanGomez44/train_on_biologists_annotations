import os,glob,sys

from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset

NO_ANNOT = -1

def preproc_annot(x):
    if x.isdigit():
        return int(x)
    else:
        return NO_ANNOT
  
def make_annot_dict(dataset_path,is_train):

    if is_train:
        annot_filename = "Gardner_train_silver.csv"
    else:
        annot_filename = "Gardner_test_gold_onlyGardnerScores.csv"

    annot_path = os.path.join(dataset_path,annot_filename)
    annot_csv = np.genfromtxt(annot_path,delimiter=";",dtype=str)

    dic = {}

    for row in annot_csv[1:]:
        sub_dic = {"exp":preproc_annot(row[1]),"icm":preproc_annot(row[2]),"te":preproc_annot(row[3])}
        #Verify dic 
        annot_nb = 0 
        for key in sub_dic:
            if sub_dic[key] != NO_ANNOT:
                annot_nb += 1 
        
        assert annot_nb>0,f"Image {row[0]} from dataset {annot_filename} has no annotation: found {annot_nb} annotation for {len(sub_dic.keys())} keys."

        dic[row[0]] = sub_dic

    return dic 

def get_transform(size, phase='train'):

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

    transf.extend([transforms.ToTensor()])
    transf.extend([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    transf = transforms.Compose(transf)

    return transf


class GradeDataset(Dataset):

    def __init__(self,dataset_path,is_train,size):

        self.is_train = is_train
        self.annot_dict = make_annot_dict(dataset_path,is_train)

        self.image_list = sorted(self.annot_dict.keys())

        self.image_fold = os.path.join(dataset_path,"Images")
        self.transf = get_transform(size, phase='train' if is_train else "eval")

    def __getitem__(self, item):

        image_name = self.image_list[item]
        image_path = os.path.join(self.image_fold,image_name)
        image = Image.open(image_path)
        image = self.transf(image)

        annot = self.annot_dict[image_name]

        return image,annot 

    def __len__(self):
        return len(self.image_list)

if __name__ == "__main__":
    import torch
    import torchvision

    for test in [True,False]:

        dataset = GradeDataset("../data/Blastocyst_Dataset",False,(224,224))

        trainLoader = torch.utils.data.DataLoader(dataset=dataset, batch_size=2,
                                                pin_memory=True, num_workers=0)


        for images,annot in trainLoader:

            print(images.shape)

            torchvision.utils.save_image(images,f"../results/grade_dataset_test={test}.png")

            for key in annot:
                print(key,annot[key])

            break