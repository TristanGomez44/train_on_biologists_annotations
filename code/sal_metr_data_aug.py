import torch
import torchvision 

from saliency_maps_metrics.multi_step_metrics import Deletion, Insertion
from saliency_maps_metrics.single_step_metrics import IIC_AD, ADD

metric_dic = {"DAUC":Deletion, "IAUC":Insertion, "AD":IIC_AD, "ADD":ADD}
metric_list = list(metric_dic.keys())
is_multi_step = {"DAUC":True, "IAUC":True, "AD":False, "ADD":False}

def get_att_maps(retDict):
    if not "attMaps" in retDict:
        attMaps = torch.abs(retDict["feat"].sum(dim=1,keepdim=True))
    else:
        attMaps = retDict["attMaps"]

    return attMaps

def apply_sal_metr_masks(model,data,mask_prob=1):

    with torch.no_grad():
        retDict = model(data)

    data_masked_list = []
    expl = get_att_maps(retDict)
    for i in range(len(data)):
        if torch.rand(size=(1,)).item() <= mask_prob:
            metric_ind = torch.randint(0,len(metric_list),size=(1,)).item()
            metric_name = metric_list[metric_ind]
            if is_multi_step[metric_name]:
                metric = metric_dic[metric_name]()

                data_i = data[i:i+1]
                masking_data_i = metric.get_masking_data(data_i)
                expl_i = expl[i:i+1]

                dic = metric.choose_data_order(data_i,masking_data_i)
                data1,data2 = dic["data1"],dic["data2"]

                k = torch.randint(0,expl.shape[2]*expl.shape[3],size=(1,)).item()
                mask,_ = metric.compute_mask(expl_i,data.shape,k)
                data_masked = metric.apply_mask(data1,data2,mask)
                data_masked_list.append(data_masked)
            else:
                metric = metric_dic[metric_name]()

                data_i = data[i:i+1]
                masking_data_i = metric.get_masking_data(data_i)

                mask = metric.compute_mask(expl[i:i+1],data.shape)
                data_masked = metric.apply_mask(data_i,masking_data_i,mask)
                data_masked_list.append(data_masked)
        else:
            data_masked_list.append(data[i:i+1])

    data_masked = torch.cat(data_masked_list,dim=0).to(data)

    return data_masked
        

        
    
        