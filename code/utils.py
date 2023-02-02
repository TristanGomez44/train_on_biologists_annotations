import torchvision
from torchvision import transforms 

inv_imgnet_norm = transforms.Compose([ transforms.Normalize(mean = [ 0., 0., 0. ],
                                                     std = [ 1/0.229, 1/0.224, 1/0.225 ]),
                                transforms.Normalize(mean = [ -0.485, -0.456, -0.406 ],
                                                     std = [ 1., 1., 1. ]),
                               ])

def save_image(img,path,**kwargs):
    img = inv_imgnet_norm(img)
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
        for i in range(len(dim)):
            tensor_min = tensor.min(dim=dim[i])[0]
            tensor_max = tensor.max(dim=dim[i])[0]

        tensor = (tensor-tensor_min)/(tensor_max-tensor_min)

    return tensor

def findNumbers(x):
    '''Extracts the numbers of a string and returns them as an integer'''

    return int((''.join(xi for xi in str(x) if xi.isdigit())))

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
