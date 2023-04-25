import os
import math 

import numpy as np
import torch
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib import colors

from args import ArgReader,init_post_hoc_arg,addLossTermArgs
import modelBuilder
import load_data
from saliency_maps_metrics.multi_step_metrics import Insertion
from does_resolution_impact_faithfulness import get_data_inds_and_explanations,load_model
import utils
def compute_or_load_scores(args,scale_factors,perturb_props,explanations,data,net_lambda):

        result_file_path = f"../results/{args.exp_id}/blur_and_pert_vs_conf_{args.model_id}_{args.att_metrics_post_hoc}.npy"
        device = data.device

        example_img_mat = []

        if not os.path.exists(result_file_path):
            data_masked_list = []
            for factor in scale_factors:
                print("\t",factor)
            
                explanations_resc = torch.nn.functional.interpolate(explanations,scale_factor=factor,mode="bicubic")            

                metric_constr_arg_dict = {}
                metric_constr_arg_dict.update({"max_step_nb":explanations.shape[2]*explanations.shape[3],"batch_size":args.val_batch_size})
                metric_constr_arg_dict.update({"cumulative":False})
                metric = Insertion(**metric_constr_arg_dict)
        
                total_pixel_nb = explanations_resc.shape[2]*explanations_resc.shape[3]

                masking_data = metric.get_masking_data(data)
                dic = metric.choose_data_order(data,masking_data)
                data1,data2 = dic["data1"],dic["data2"]

                for perturbation_prop in perturb_props:
                    
                    perturbation_nb = int(total_pixel_nb*perturbation_prop)

                    batch_img = []
                    for i in range(len(data)):

                        mask,_ = metric.compute_mask(explanations_resc[i:i+1],data1.shape,total_pixel_nb//2+perturbation_nb//2,perturbation_nb)
                        mask = mask.to(data1.device)
                        data_masked = metric.apply_mask(data1[i:i+1],data2[i:i+1],mask)
                        batch_img.append(data_masked)

                        if i == 0:
                            example_img_mat.append(data_masked)

                    batch_img = torch.cat(batch_img,dim=0).cpu()
                    data_masked_list.append(batch_img)

            data_masked_list = torch.cat(data_masked_list,dim=0)
            data_masked_chunks = torch.split(data_masked_list,args.val_batch_size)   
            output_list = []
            for data_masked in data_masked_chunks:
                output = net_lambda(data_masked.to(device))           
                output_list.append(output)
            
            output_list = torch.cat(output_list,dim=0)
            torch.save(output_list,result_file_path)

            example_img_mat = torch.cat(example_img_mat,dim=0)
            utils.save_image(example_img_mat,f"../vis/{args.exp_id}/blur_and_pert_vs_conf_examples_{args.model_id}_{args.att_metrics_post_hoc}.png",row_nb=len(scale_factors))

        else:
            output_list = torch.load(result_file_path)

        print(output_list.shape,len(scale_factors),args.nb_pert,len(data),-1)
        output_list = output_list.reshape(len(scale_factors),args.nb_pert,len(data),-1)

        return output_list
            
def blur_data(data,kernel_size):
    kernel = torch.ones(kernel_size,kernel_size)
    kernel = kernel/kernel.numel()
    kernel = kernel.unsqueeze(0).unsqueeze(0).expand(3,1,-1,-1)
    kernel = kernel.to(data.device)
    data = F.conv2d(data,kernel,padding=kernel.size(-1)//2,groups=kernel.size(0))  
    return data

def main(argv=None):

    #Getting arguments from config file and command line
    #Building the arg reader
    argreader = ArgReader(argv)
    argreader = load_data.addArgs(argreader)
    argreader = modelBuilder.addArgs(argreader)
    argreader = init_post_hoc_arg(argreader)
    argreader.parser.add_argument('--kernel_size', type=int,default=121)
    argreader.parser.add_argument('--max_pert_prop', type=float,default=0.5)
    argreader.parser.add_argument('--nb_pert', type=int,default=5)
    argreader = addLossTermArgs(argreader)

    #Reading the comand line arg
    argreader.getRemainingArgs()

    #Getting the args from command line and config file
    args = argreader.args

    _,testDataset = load_data.buildTestLoader(args, "test")

    net,net_lambda = load_model(args)

    data,explanations,predClassInds,outputs,_ = get_data_inds_and_explanations(net,net_lambda,testDataset,args)
    batch_inds = torch.arange(len(data))

    if "ablationcam" in args.att_metrics_post_hoc:
        net,net_lambda = load_model(args)

    outputs = torch.softmax(outputs,dim=-1)[batch_inds,predClassInds]

    kernel_size = args.kernel_size

    data_blurred = blur_data(data,kernel_size)
    outputs_blurred = net_lambda(data_blurred)
    outputs_blurred = torch.softmax(outputs_blurred,dim=-1)[batch_inds,predClassInds]

    ratio = net.firstModel.featMod.downsample_ratio
    powers = np.arange(math.log(ratio,2)+1)
    scale_factors = np.power(2,powers).astype(int)
    #scale_factors = scale_factors[:4]

    log = int(math.log(1/args.max_pert_prop,2))
    log_list = np.arange(log,log+args.nb_pert)
    perturb_props = 1/np.power(2,log_list).astype(int)

    output_list = compute_or_load_scores(args,scale_factors,perturb_props,explanations,data,net_lambda)
    output_list = torch.softmax(output_list,dim=-1)

    val_matrix = torch.zeros(len(scale_factors),args.nb_pert)
    batch_inds = torch.arange(len(data))
    for i in range(len(scale_factors)):
        for j in range(args.nb_pert):
            val_matrix[i,j] = output_list[i,j,batch_inds,predClassInds].mean()

    fontsize = 17
    cmap = plt.get_cmap('plasma')
    
    plt.figure()
    plt.imshow(val_matrix,cmap=cmap)

    for i in range(val_matrix.shape[0]):
        for j in range(val_matrix.shape[1]):
            plt.text(j,i+0.1,round(val_matrix[i,j].item()*100),ha="center",fontsize=fontsize) 

    cbar = plt.colorbar(cm.ScalarMappable(norm=colors.Normalize(val_matrix.min(),val_matrix.max()),cmap=cmap))
    cbar.ax.tick_params(labelsize=fontsize)

    plt.ylabel("Resolution",fontsize=fontsize)
    plt.yticks(np.arange(len(scale_factors)),explanations.shape[2]*scale_factors,fontsize=fontsize)
    
    plt.xlabel("Unperturbed proportion",fontsize=fontsize)
    xticks = (1/perturb_props).astype(int).astype(str)
    xticks = ["1/"+tick for tick in xticks]
    plt.xticks(np.arange(len(perturb_props)),xticks,fontsize=fontsize)

    plt.tight_layout()
    plt.savefig(f"../vis/{args.exp_id}/blur_and_pert_vs_conf_{args.model_id}_{args.att_metrics_post_hoc}.png")

if __name__ == "__main__":
    main()