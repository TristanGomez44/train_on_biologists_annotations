import argparse
import json
import glob
import collections
import os

from PIL import Image 
import numpy as np 
import torch

import enums
from format_data_phase_two import add_annot

def get_output(args):
    train_images_dic = {}
    for i,split_file in enumerate([args.split_file_1,args.split_file_2]):
        with open(split_file) as f:
            train_images_dic[i] = json.load(f)["train"]

    output_dic = {}
    image_name_to_output = collections.defaultdict(lambda :{})
    for task in enums.Tasks:
        output_dic[task] = {}

        for i,model_id in enumerate([args.model_id_1,args.model_id_2]):

            output_path = glob.glob(f"../results/{args.exp_id}/output_{task.value}_{model_id}_epoch*_train.npy")[0]

            output = torch.tensor(np.load(output_path))
            output = torch.softmax(output,dim=-1)
            confidence,prediction = output.max(dim=-1)

            for j in range(len(output)):
                prediction_i = list(enums.annot_enum_dic[task])[prediction[i].item()].value

                image_name_to_output[train_images_dic[i][j]][task] = {"confidence":confidence[i],"prediction":prediction_i}

    return image_name_to_output


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--split_file_1",type=str)
    parser.add_argument("--split_file_2",type=str)
    parser.add_argument("--model_id_1",type=str)    
    parser.add_argument("--model_id_2",type=str)   
    parser.add_argument("--exp_id",type=str,default="GRADES2")           
    parser.add_argument("--out_fold_path",type=str)   
    parser.add_argument("--image_fold_path",type=str)       
    parser.add_argument('--low_conf_thres',type=int,default=0.45)     
    parser.add_argument('--high_conf_thres',type=int,default=0.7)     
    args = parser.parse_args()

    image_name_to_output = get_output(args)

    os.makedirs(args.out_fold_path,exist_ok=True)

    for image_name in image_name_to_output.keys():
        
        image_suff = image_name.replace("F0","")
        img_paths = glob.glob(os.path.join(args.image_fold_path,"*"+image_suff))
        for path in img_paths:

            image = Image.open(path)

            subdic = image_name_to_output[image_name]

            te = subdic[enums.Tasks.TE]["prediction"]
            icm = subdic[enums.Tasks.ICM]["prediction"]
            exp = subdic[enums.Tasks.EXP]["prediction"]

            te_conf = subdic[enums.Tasks.TE]["confidence"]
            icm_conf = subdic[enums.Tasks.ICM]["confidence"]
            exp_conf = subdic[enums.Tasks.EXP]["confidence"]

            original_size = image.size

            image_with_annot = add_annot(image,te,icm,exp,te_conf,icm_conf,exp_conf,args.low_conf_thres,args.high_conf_thres)

            image_with_annot = image_with_annot.resize(original_size, Image.Resampling.LANCZOS)

            filename = os.path.basename(path)
            image_with_annot.save(os.path.join(args.out_fold_path,filename))

if __name__ == "__main__":
    main()