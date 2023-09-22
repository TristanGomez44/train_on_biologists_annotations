
import os
import zipfile
import shutil
import json

import numpy as np
import pandas as pd

from args import ArgReader

def format_dataset(path_to_zip,path_to_annot_csv,dest_folder,keys,split_sizes,fold_name="blastocyst_dataset",json_file_name="splits.json",seed=0):

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
	for key in keys:
		aggr_annotations = grouped[key].apply(lambda x: x.mode().iloc[0])
		aggr_annotations = aggr_annotations.reset_index()

		if all_aggr_annot is None:
			all_aggr_annot = aggr_annotations
		else:
			all_aggr_annot[key] = aggr_annotations[key]
	
	path_to_aggr_annot_csv = os.path.join(dest_folder,"aggregated_annotations.csv")

	all_aggr_annot.to_csv(path_to_aggr_annot_csv)
	
	splits = make_split(grouped,split_sizes,seed)

	json_file_path = os.path.join(dest_folder,json_file_name)
	with open(json_file_path, 'w') as fp:
		json.dump(splits, fp)

def make_split(grouped,split_sizes,seed=0):

	img_names = sorted([img_name for (img_name,_) in list(grouped["image_name"])])
	
	img_names = np.array(img_names)
	np.random.seed(seed)
	np.random.shuffle(img_names)

	train_size = int(len(img_names)*split_sizes[0])
	val_size = int(len(img_names)*(split_sizes[1]))

	train_set = img_names[:train_size].tolist()
	val_set = img_names[train_size:train_size+val_size].tolist()
	test_set = img_names[train_size+val_size:].tolist()

	return {"train":train_set,"val":val_set,"test":test_set}

def main(argv=None):

	#Getting arguments from config file and command row
	#Building the arg reader
	argreader = ArgReader(argv)

	argreader.parser.add_argument('--path_to_zip',type=str)
	argreader.parser.add_argument('--path_to_annot_csv',type=str)
	argreader.parser.add_argument('--dest_folder',type=str,default="../data/dl4ivf_blastocysts/")
	argreader.parser.add_argument('--keys',type=str,nargs="*",default=["icm","te","exp"])
	argreader.parser.add_argument('--split_sizes',type=float,nargs=2,default=[0.5,0.25],help="Only indicates the proportions of the train and validation set. Test set size is deduced automatically.")

	#Reading the comand row arg
	argreader.getRemainingArgs()

	#Getting the args from command row and config file
	args = argreader.args

	format_dataset(args.path_to_zip,args.path_to_annot_csv,args.dest_folder,args.keys,args.split_sizes)
if __name__ == "__main__":
    main()
