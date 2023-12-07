
import os
import glob

import torchvision
from torchvision import transforms 
import torch 
import numpy as np

from grade_dataset import NO_ANNOT
import init_model
import modelBuilder
import load_data
from enums import Tasks
inv_imgnet_norm = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

def make_class_nb_dic(args):
    return {"ICM":args.icm_te_class_nb,"TE":args.icm_te_class_nb,"EXP":args.exp_class_nb}

def _remove_no_annot(tensor,reference):
    return tensor[reference!=NO_ANNOT]

def remove_no_annot(output,target):
    output = _remove_no_annot(output,reference=target)
    target = _remove_no_annot(target,reference=target)
    return output,target

def make_grid(img,row_nb):
    assert len(img) % row_nb == 0
    col_nb = len(img)//row_nb 
    img = img.reshape(row_nb,col_nb,img.shape[1],img.shape[2],img.shape[3])

    grid = []
    for i in range(row_nb):
        row = []
        for j in range(col_nb):
            row.append(img[i,j])
        
        row = torch.cat(row,dim=2)
        grid.append(row)

    grid = torch.cat(grid,dim=1)
    grid = grid.unsqueeze(0)

    return grid

def save_image(img,path,mask=None,row_nb=None,apply_inv_norm=True,**kwargs):
    
    if img.shape[1] == 3 and apply_inv_norm:
        if mask is None:
            mask = (img!=0)
        img = inv_imgnet_norm(img)*mask
    
    if not row_nb is None:
        img = make_grid(img,row_nb)

    torchvision.utils.save_image(img,path,**kwargs)

class Bunch(object):
    def __init__(self, adict):
        self.__dict__.update(adict)

def normalize_tensor(tensor,dim=None):

    if dim is None:
        tensor = (tensor-tensor.min())/(tensor.max()-tensor.min())
    else:
        tensor_min = tensor
        tensor_max = tensor
        for _ in range(len(dim)):
            tensor_min = tensor_min.min(dim=-1)[0]
            tensor_max = tensor_max.max(dim=-1)[0]
        tensor_min = tensor_min.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        tensor_max = tensor_max.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        tensor = (tensor-tensor_min)/(tensor_max-tensor_min)

    return tensor

def findNumbers(x):
    '''Extracts the numbers of a string and returns them as an integer'''

    return int((''.join(xi for xi in str(x) if xi.isdigit())))

def getEpoch(path):
    return int(os.path.basename(path).split("epoch")[1].split("_")[0])

def findLastNumbers(weightFileName):
    '''Extract the epoch number of a weith file name.

    Extract the epoch number in a weight file which name will be like : "clustDetectNet2_epoch45".
    If this string if fed in this function, it will return the integer 45.

    Args:
        weightFileName (string): the weight file name
    Returns: the epoch number

    '''

    i=0
    res = ""
    allSeqFound = False
    while i<len(weightFileName) and not allSeqFound:
        if not weightFileName[len(weightFileName)-i-1].isdigit():
            allSeqFound = True
        else:
            res += weightFileName[len(weightFileName)-i-1]
        i+=1

    res = res[::-1]

    return int(res)

def get_datasets_and_model(args):
    train_loader,train_dataset = load_data.buildTestLoader(args,"train")
    test_loader,test_dataset = load_data.buildTestLoader(args,"test")

    net = modelBuilder.netBuilder(args)
    best_paths = glob.glob(f"../models/{args.exp_id}/model{args.model_id}_best_epoch*")
    assert len(best_paths) == 1, "There should be only one best model"
    best_path = best_paths[0]
    net = init_model.preprocessAndLoadParams(best_path,args.cuda,net,ssl=args.ssl)
    net.eval()
    torch.set_grad_enabled(False)

    return train_loader,train_dataset,test_loader,test_dataset,net 

def get_feature_and_output(train_loader,test_loader,net,args):
    result_dic = {}

    if args.task_to_train == "all":
        ind = [task.value for task in Tasks].index(args.task_output_to_save)

    for subset,loader in zip(["train","test"],[train_loader,test_loader]):
        pooled_features_path = f"../results/{args.exp_id}/pooled_features_{args.model_id}_{subset}.npy"
        
        result_dic[subset] = {}
        if not os.path.exists(pooled_features_path):
            print(f"Inference on {subset} set")
            pooled_features = []
            features = []
            output = []
            target = []
            for idx,batch in enumerate(loader):
        
                if idx%10 == 0:
                    print(f"\tProcessing batch {idx}/{len(loader)}")

                image,annot_dict = batch
                output_dic = net(image)        

                pooled_features_tmp = output_dic["feat_pooled"].cpu().numpy()
                if args.task_to_train == "all" and args.resnet_bilinear:
                    pooled_features.append(pooled_features_tmp[:,ind])
                else:
                    pooled_features.append(pooled_features_tmp)

                features.append(output_dic["feat"].cpu().numpy())                    
                output.append(output_dic["output_"+args.task_output_to_save].cpu().numpy())
                target.append(annot_dict[args.task_output_to_save].cpu().numpy())

                if args.debug:
                    break

            pooled_features = np.concatenate(pooled_features,axis=0)
            features = np.concatenate(features,axis=0)
            output = np.concatenate(output,axis=0)
            target = np.concatenate(target,axis=0)

            np.save(pooled_features_path,pooled_features)
            np.save(pooled_features_path.replace("pooled_features","features"),features)
            np.save(pooled_features_path.replace("pooled_features","output"),output)
            np.save(pooled_features_path.replace("pooled_features","target"),target)

        else:
            pooled_features = np.load(pooled_features_path)
            features = np.load(pooled_features_path.replace("pooled_features","features"))
            output = np.load(pooled_features_path.replace("pooled_features","output"))
            target = np.load(pooled_features_path.replace("pooled_features","target"))

        result_dic[subset]["pooled_features"] = pooled_features
        result_dic[subset]["features"] = features
        result_dic[subset]["output"] = output
        result_dic[subset]["target"] = target
    
    return result_dic
