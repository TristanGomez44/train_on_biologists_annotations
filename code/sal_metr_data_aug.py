import torch
import torchvision 

from saliency_maps_metrics.multi_step_metrics import Deletion, Insertion
from saliency_maps_metrics.single_step_metrics import IIC_AD, ADD

metric_dic = {"DAUC":Deletion, "IAUC":Insertion, "AD":IIC_AD, "ADD":ADD}
metric_list = list(metric_dic.keys())
is_multi_step = {"DAUC":True, "IAUC":True, "AD":False, "ADD":False}
is_masking_object = {"DAUC":True,"IAUC":False,"AD":False,"ADD":True}

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
    is_masking_object_list = []
    for i in range(len(data)):
        if torch.rand(size=(1,)).item() <= mask_prob:
            metric_ind = torch.randint(0,len(metric_list),size=(1,)).item()
            metric_name = metric_list[metric_ind]
            is_masking_object_list.append(is_masking_object[metric_name])
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
            is_masking_object_list.append(False)

    data_masked = torch.cat(data_masked_list,dim=0).to(data)

    return data_masked,is_masking_object_list
        
def apply_sal_metr_masks_and_update_dic(model,data,args,resDict):

    data_masked,is_object_masked_list = apply_sal_metr_masks(model,data,args.sal_metr_mask_prob)
    resDict["is_object_masked_list"] = is_object_masked_list
    if args.nce_weight > 0 or args.adv_weight > 0 or args.focal_loss_on_masked:
        resDict_masked = model(data_masked) 
        resDict.update({key+"_masked":resDict_masked[key] for key in resDict_masked})
    else:
        data = data_masked
    return resDict,data
        
    
        