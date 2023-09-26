
import os,sys
import zipfile
import shutil
import json

import numpy as np
import pandas as pd

from args import ArgReader

from grade_dataset import Tasks

def format_dl4ivf_dataset(path_to_zip,path_to_annot_csv,dest_folder,train_prop,val_prop,fold_name="blastocyst_dataset",json_file_name="splits.json",seed=0):

	os.makedirs(dest_folder,exist_ok=True)

	zip_name = os.path.splitext(os.path.basename(path_to_zip))[0]
	extracted_dataset_path = os.path.join(dest_folder,zip_name)

	if not os.path.exists(extracted_dataset_path):
		with zipfile.ZipFile(path_to_zip, 'r') as zip_ref:
			zip_ref.extractall(dest_folder)

	img_folder_path = os.path.join(dest_folder,fold_name)
	new_img_folder_path = img_folder_path.replace(fold_name,"Images")
	if not os.path.exists(new_img_folder_path):
		os.rename(img_folder_path,new_img_folder_path)

	csv_file_name = os.path.splitext(os.path.basename(path_to_annot_csv))[0]

	new_path_to_annot_csv = os.path.join(dest_folder,csv_file_name)
	shutil.copy(path_to_annot_csv,new_path_to_annot_csv)

	annotation_df = pd.read_csv(new_path_to_annot_csv)
	grouped = annotation_df.groupby('image_name')

	all_aggr_annot = None
	for task in Tasks:
		task_name = task.value
		aggr_annotations = grouped[task_name].apply(lambda x: x.mode().iloc[0])
		aggr_annotations = aggr_annotations.reset_index()

		if all_aggr_annot is None:
			all_aggr_annot = aggr_annotations
		else:
			all_aggr_annot[task_name] = aggr_annotations[task_name]
	
	path_to_aggr_annot_csv = os.path.join(dest_folder,"aggregated_annotations.csv")

	all_aggr_annot.to_csv(path_to_aggr_annot_csv,index=False)
	
	splits = make_split(all_aggr_annot,train_prop,val_prop,seed)

	json_file_path = os.path.join(dest_folder,json_file_name)
	with open(json_file_path, 'w') as fp:
		json.dump(splits, fp)

def make_split(all_aggr_annot,train_prop,val_prop,seed=0):

	groups = all_aggr_annot.groupby([task.value for task in Tasks])
	
	np.random.seed(seed)

	train_set,val_set,test_set = [],[],[]

	for (_,sub_df) in groups:
		
		img_names = list(sub_df["image_name"])
		img_names = np.array(img_names)

		np.random.shuffle(img_names)

		train_size = round(len(img_names)*train_prop)
		val_size = round(len(img_names)*val_prop)

		train_set.extend(img_names[:train_size].tolist())
		val_set.extend(img_names[train_size:train_size+val_size].tolist())
		test_set.extend(img_names[train_size+val_size:])

	print(len(train_set),len(val_set),len(test_set))

	return {"train":sorted(train_set),"val":sorted(val_set),"test":sorted(test_set)}

def main(argv=None):

	#Getting arguments from config file and command row
	#Building the arg reader
	argreader = ArgReader(argv)

	argreader.parser.add_argument('--path_to_zip',type=str)
	argreader.parser.add_argument('--path_to_annot_csv',type=str)
	argreader.parser.add_argument('--dest_folder',type=str,default="../data/dl4ivf_blastocysts/")
	argreader.parser.add_argument('--train_prop',type=float,default=0.4)
	argreader.parser.add_argument('--val_prop',type=float,default=0.1)

	#Reading the comand row arg
	argreader.getRemainingArgs()

	#Getting the args from command row and config file
	args = argreader.args

	format_dl4ivf_dataset(args.path_to_zip,args.path_to_annot_csv,args.dest_folder,args.train_prop,args.val_prop)

if __name__ == "__main__":
    main()
