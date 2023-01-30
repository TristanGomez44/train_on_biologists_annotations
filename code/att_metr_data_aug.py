import torch

from saliency_maps_metrics_tristangomez44.multi_step_metrics import DAUC, IAUC
from saliency_maps_metrics_tristangomez44.single_step_metrics import AD, ADD

metric_dic = {"DAUC":DAUC, "IAUC":IAUC, "AD":AD, "ADD":ADD}
metric_list = list(metric_dic.keys())
is_multi_step = {"DAUC":True, "IAUC":True, "AD":False, "ADD":False}

def get_att_maps(retDict):
    if not "attMaps" in retDict:
        attMaps = torch.abs(retDict["feat"].sum(dim=1,keepdim=True))
    else:
        attMaps = retDict["attMaps"]

    return attMaps

def apply_att_metr_masks(model,data):

    with torch.no_grad():
        retDict = model(data)

    data_masked_list = []
    expl = get_att_maps(retDict)
    for i in range(len(data)):
        metric_ind = torch.randint(0,len(metric_list),size=(1,)).item()
        metric_name = metric_list[metric_ind]
        if is_multi_step[metric_name]:
            metric = metric_dic[metric_name](data.shape,expl.shape,True)

            data_to_replace_with_i = metric.init_data_to_replace_with(data[i:i+1])
            data_i = metric.preprocess_data(data[i:i+1]) 
            expl_i = expl[i:i+1]

            k = torch.randint(0,expl.shape[2]*expl.shape[3],size=(1,)).item()
            mask,_ = metric.compute_mask(expl_i,data.shape,k)
            data_masked = metric.apply_mask(data_i,data_to_replace_with_i,mask)
        else:
            metric = metric_dic[metric_name]()
            mask = metric.compute_mask(expl[i:i+1],data.shape)
            data_masked = metric.apply_mask(data[i:i+1],mask)

        data_masked_list.append(data_masked)
        
    data_masked = torch.cat(data_masked_list,dim=0).to(data)

    return data_masked
        

        
    
        