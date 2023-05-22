
import os
import torchvision
from torchvision import transforms 
import torch 
from grade_dataset import NO_ANNOT

inv_imgnet_norm = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

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

def save_image(img,path,mask=None,row_nb=None,**kwargs):
    
    if img.shape[1] == 3:
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
