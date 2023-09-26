
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

def make_split(all_aggr_annot,train_prop,val_prop=None,seed=0):

	groups = all_aggr_annot.groupby([task.value for task in Tasks])
	
	np.random.seed(seed)

	train_set,val_set,test_set = [],[],[]

	for (_,sub_df) in groups:
		
		img_names = list(sub_df["image_name"])
		img_names = np.array(img_names)

		np.random.shuffle(img_names)

		train_size = round(len(img_names)*train_prop)
		if val_prop is None:
			val_size = len(img_names) - train_size
		else:
			val_size = round(len(img_names)*val_prop)

		train_set.extend(img_names[:train_size].tolist())
		val_set.extend(img_names[train_size:train_size+val_size].tolist())
		test_set.extend(img_names[train_size+val_size:])

	return {"train":sorted(train_set),"val":sorted(val_set),"test":sorted(test_set)}

def get_match_between_csv_and_tasks(columns):
	column_match_dic = {}

	for task in Tasks:
		matching_col_found = False
		col_ind = 0
		while not matching_col_found and col_ind < len(columns):

			if task.value in columns[col_ind]:
				matching_col_found=True
				column_match_dic[columns[col_ind]] = task.value

			col_ind += 1
		
		if not matching_col_found:
			raise ValueError("Could not find match between multicenter train csv columns and tasks.")
	return column_match_dic

def make_multicenter_split(path_to_dataset,train_prop,json_file_name="splits.json"):

	path_to_train_annot= os.path.join(path_to_dataset,"Gardner_train_silver.csv")
	path_to_test_annot = os.path.join(path_to_dataset,"Gardner_test_gold_onlyGardnerScores.csv")

	train_csv = pd.read_csv(path_to_train_annot,delimiter=";")

	column_match_dic = get_match_between_csv_and_tasks(train_csv.columns)
	column_match_dic["Image"] = "image_name"

	train_csv = train_csv.rename(column_match_dic,axis=1)

	splits = make_split(train_csv,train_prop,val_prop=None,seed=0)
	splits["test"] = list(pd.read_csv(path_to_test_annot,delimiter=";")["Image"])

	dest_folder = os.path.dirname(path_to_train_annot)
	json_file_path = os.path.join(dest_folder,json_file_name)

	with open(json_file_path, 'w') as fp:
		json.dump(splits, fp)

def main(argv=None):

	#Getting arguments from config file and command row
	#Building the arg reader
	argreader = ArgReader(argv)

	argreader.parser.add_argument('--format_dl4ivf_dataset',action="store_true")
	argreader.parser.add_argument('--path_to_zip',type=str)
	argreader.parser.add_argument('--path_to_dl4ivf_annot',type=str)
	argreader.parser.add_argument('--dest_folder',type=str,default="../data/dl4ivf_blastocysts/")

	argreader.parser.add_argument('--format_multicenter_dataset',action="store_true")
	argreader.parser.add_argument('--path_to_multicenter_dataset',type=str)

	argreader.parser.add_argument('--train_prop',type=float,default=0.4)
	argreader.parser.add_argument('--val_prop',type=float,default=0.1)

	#Reading the comand row arg
	argreader.getRemainingArgs()

	#Getting the args from command row and config file
	args = argreader.args

	assert args.format_dl4ivf_dataset or args.format_multicenter_dataset,"Choose one of the two options"

	if args.format_dl4ivf_dataset:
		format_dl4ivf_dataset(args.path_to_zip,args.path_to_dl4ivf_annot,args.dest_folder,args.train_prop,args.val_prop)
	else:
		make_multicenter_split(args.path_to_multicenter_dataset,args.train_prop)
if __name__ == "__main__":
    main()
