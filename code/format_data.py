
import os
import zipfile
import shutil
import json
import subprocess 
from collections import defaultdict,Counter

import numpy as np
import pandas as pd

from args import ArgReader

from enums import annot_enum_dic
from grade_dataset import Tasks

from ZREC_Mos_Recovery import zrec_mos_recovery

def normalize(subject_inconsistency):
	min_incons = subject_inconsistency.min()
	max_incons = subject_inconsistency.max()
	return (subject_inconsistency-min_incons)/(max_incons-min_incons)

def zmos_recovery(image_names,dest_folder):

	annotations = pd.read_csv(os.path.join(dest_folder,"database.csv"),dtype=str,keep_default_na=False)
	annotations.reset_index()

	idUsers = sorted(list(set(annotations["idUser"])))
	image_names = sorted(image_names)

	idUser_dic = {idUsers[i]:i for i in range(len(idUsers))}
	nameImage_dic = {image_names[i]:i for i in range(len(image_names))}	

	matrix_dic = {task:np.nan*np.ones((len(idUsers),len(image_names))) for task in Tasks}

	annot_enum_fmt_dic = {task:[str(annot.value) for annot in annot_enum_dic[task]] for task in Tasks}

	for _,(image_name,annot,idUser) in annotations.iterrows():
		
		task,value = get_task_and_value(annot)

		if task is not None:
			task = Tasks(task)		
			value = list(annot_enum_fmt_dic[task]).index(value)+1
			matrix_dic[task][idUser_dic[idUser]][nameImage_dic[image_name]] = value

	zmos_col_list = [image_names]
	pondered_vote_col_list = [image_names]
	conf_col_list = [image_names]
	for task in Tasks:

		mos_scores,subject_inconsistency = zrec_mos_recovery(matrix_dic[task],dest_folder)
		mos_scores -= 1
		
		print(task,matrix_dic[task].shape,mos_scores.shape,subject_inconsistency.shape)
		
		norm_consistency = 1-normalize(subject_inconsistency)

		possible_values = list(annot_enum_dic[task])
		possible_values_class_inds = [possible_values.index(value)+1 for value in possible_values]

		pondered_votes = []
		confidences = []
		for j in range(len(mos_scores)):
			weights = []
			for value in possible_values_class_inds:		
				weights.append(norm_consistency[matrix_dic[task][:,j]==value].sum())
			weights = np.array(weights)

			pondered_votes.append(possible_values[np.argmax(weights)].value)
			confidences.append(np.max(weights)/np.sum(weights))

		pondered_votes = np.array(pondered_votes)
		confidences = np.array(confidences)

		zmos_col_list.append(mos_scores)
		pondered_vote_col_list.append(pondered_votes)
		conf_col_list.append(confidences)

	for col_list,suff in zip([zmos_col_list,pondered_vote_col_list,conf_col_list],["ZRECMOS","PONDERED_VOTE","CONFIDENCE"]):

		col_list = np.stack(col_list,axis=1)

		csv = pd.DataFrame(data=col_list,columns=["image_name"]+[task.value for task in Tasks])

		csv.to_csv(os.path.join(dest_folder,f"aggregated_annotations_{suff}.csv"),index=False,sep=" ")

def get_task_and_value(annot):

	assert len(annot) in [3,6],f"Annot has incorect length:{annot}"

	pref,value = annot[:2],annot[2:]

	assert pref in ["Tr","Bo","Ex","Na"],f"Inccorect annotation prefix:{pref}"

	if pref =="Tr":
		task = Tasks.TE
	elif pref == "Bo":
		task = Tasks.ICM
	elif pref== "Ex":
		task = Tasks.EXP
	else:
		task = None

	return task,value

def convert_db_to_csv(new_path_to_annot_file):

	csv_filename = os.path.splitext(os.path.basename(new_path_to_annot_file))[0]+".csv"
	csv_dirname = os.path.dirname(new_path_to_annot_file)
	csv_path = os.path.join(csv_dirname,csv_filename)

	queries = ["sqlite3",new_path_to_annot_file,".mode csv",".headers on",f".output {csv_path}","select nameImage,idTag,idUser from annotation join image on annotation.idImg=image.id join user on user.id==annotation.idUser where username != 'debug'"]

	subprocess.call(queries)

	return csv_path

def aggregate_annotations(new_path_to_annot_file):
	with open(new_path_to_annot_file) as file:
		rows = [line.rstrip() for line in file]
	all_annot_dic = defaultdict(lambda:{task:[] for task in Tasks})
	aggr_annot = []
	distr_aggr_annot = defaultdict(lambda :[])
	for row in rows[1:]:
		image_name,annot,_ = row.split(",")
		task,value = get_task_and_value(annot)
		if task is not None:
			all_annot_dic[image_name][task].append(value)

	for image_name in sorted(all_annot_dic.keys()):
		csv_row = [image_name]
		for task in Tasks:
			annot_list = all_annot_dic[image_name][task]
			#annot_list = list(map(lambda x:x if x!="NaN" else NO_ANNOT,annot_list))

			possible_annot_values = list(annot_enum_dic[task])

			distr_row = [image_name]

			counter = Counter(annot_list)
			csv_row.append(counter.most_common(1)[0][0])
			
			for value in possible_annot_values:
				distr_row.append(counter[str(value.value)])
			
			distr_aggr_annot[task].append(distr_row)

		aggr_annot.append(csv_row)

	return aggr_annot,distr_aggr_annot,all_annot_dic.keys()

def get_header():
	return ["image_name"]+[task.value for task in Tasks]

def get_distr_csv_header(task):
	possible_values = list(annot_enum_dic[task])
	return ["image_name"]+[str(value.value) for value in possible_values]

def format_annotations(path_to_annot_file,dest_folder):
	annot_file_name = os.path.basename(path_to_annot_file)
	new_path_to_annot_file = os.path.join(dest_folder,annot_file_name)
	shutil.copy(path_to_annot_file,new_path_to_annot_file)
	new_path_to_annot_file = convert_db_to_csv(new_path_to_annot_file)	
	aggr_annot,distr_aggr_annot,img_names = aggregate_annotations(new_path_to_annot_file)
	path_to_aggr_annot_csv = os.path.join(dest_folder,"aggregated_annotations.csv")
	header = get_header()
	np.savetxt(path_to_aggr_annot_csv,aggr_annot,header=" ".join(header),fmt="%s",comments='')

	for task in Tasks:
		task_header = get_distr_csv_header(task)
		np.savetxt(path_to_aggr_annot_csv.replace(".csv","_"+task.value+".csv"),distr_aggr_annot[task],fmt="%s",comments='',header=" ".join(task_header))

	return aggr_annot,header,img_names

def format_dl4ivf_dataset(path_to_zip,path_to_annot_file,dest_folder,train_prop,val_prop,fold_name="blastocyst_dataset",json_file_name="splits.json",seed=0,train_last=False):

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

	aggr_annot,header,img_names = format_annotations(path_to_annot_file,dest_folder)

	zmos_recovery(img_names,dest_folder)

	splits = make_split(aggr_annot,header,train_prop,val_prop,seed,train_last=train_last)

	json_file_path = os.path.join(dest_folder,json_file_name)
	with open(json_file_path, 'w') as fp:
		json.dump(splits, fp)

def make_split(all_aggr_annot,headers,train_prop,val_prop=None,seed=0,train_last=False):

	all_aggr_annot = pd.DataFrame(all_aggr_annot,columns=headers)
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

		if train_last:
			test_size = len(img_names) - train_size - val_size
			val_set.extend(img_names[:val_size].tolist())
			test_set.extend(img_names[val_size:val_size+test_size])
			train_set.extend(img_names[val_size+test_size:].tolist())
		else:
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

	#Deleting duplicate annotation 
	inds = train_csv.index[train_csv['image_name'] == "838_02.png"]
	train_csv = train_csv.drop(inds[1])

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
	argreader.parser.add_argument('--train_last',action="store_true")

	argreader.parser.add_argument('--format_multicenter_dataset',action="store_true")
	argreader.parser.add_argument('--path_to_multicenter_dataset',type=str)

	argreader.parser.add_argument('--train_prop',type=float,default=0.4)
	argreader.parser.add_argument('--val_prop',type=float,default=0.1)

	#Reading the comand row arg
	argreader.getRemainingArgs()

	#Getting the args from command row and config file
	args = argreader.args

	assert args.format_dl4ivf_dataset or args.format_multicenter_dataset,"Choose one of the two options"

	assert os.path.splitext(args.path_to_dl4ivf_annot)[1] == ".db",'Annotations should be in database format'

	if args.format_dl4ivf_dataset:
		format_dl4ivf_dataset(args.path_to_zip,args.path_to_dl4ivf_annot,args.dest_folder,args.train_prop,args.val_prop,train_last=args.train_last)
	else:
		make_multicenter_split(args.path_to_multicenter_dataset,args.train_prop)

if __name__ == "__main__":
    main()
