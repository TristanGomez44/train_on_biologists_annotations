import glob, os, sys, shutil

import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from PIL import Image,ImageEnhance,ImageFile,UnidentifiedImageError
ImageFile.LOAD_TRUNCATED_IMAGES = True
from skimage import transform
import cv2
import subprocess

def make_frame_and_plane_dict(root):

    dic = {}
    for filename in ["blast_and_grade_annot_framefocal.csv","blast_and_no_grade_annot_framefocal.csv"]:
        df = pd.read_csv(root+"/"+filename,delimiter=",")

        for _, row in df.iterrows():
            name,original_frame_ind = row["# names"].split("_RUN")
            name = name.replace("_WELL","-")

            if not math.isnan(row["frame_to_use"]):

                if name in dic:
                    raise ValueError("Video is already in dictionnary.",filename,name)

                dic[name] = {"frame_to_use":int(row["frame_to_use"])}

                #Check if focal_plane value is NaN, i.e. empty.
                if type(row["focal_plane"]) is not float:
                    dic[name]["focal_plane"] = row["focal_plane"]
            else:
                original_frame_ind = original_frame_ind.replace(".jpeg","")
                dic[name] = {"frame_to_use":int(original_frame_ind),"focal_plane":"F0"}

    return dic         

def fix_annot_files(fold):

    csv = np.genfromtxt(fold[:-1]+".csv",delimiter=",",dtype=str)
    new_csv = []

    for row in csv[1:]:

        name = row[0].replace(".jpeg","")

        pattern = fold+"/"+name+"_RUN*.jpeg"

        matches = glob.glob(pattern)

        if len(matches) != 1:
            raise ValueError("Wrong matching path number",matches,pattern,row)
        
        fullname = os.path.basename(matches[0])

        new_row = [fullname]
        new_row += list(row[1:])

        new_csv.append(new_row)

    header = ",".join(csv[0])
    np.savetxt(fold[:-1]+"_fixed.csv",new_csv,header=header,delimiter=",",fmt='%s')


def str_to_float(var):
    if type(var) is str:
        return float(var.replace(",","."))
    else:
        return var

def get_img_nb(path):
    img_nb = len(glob.glob(os.path.join(path,"F0","*.*")))
    return img_nb
    
def get_prop(row,time):

    if time == -1:
        prop = 1
    else:
        end = str_to_float(row["End Time"])
        start = str_to_float(row["Start Time"])

        total_time = (end-start)*24
        prop = time/total_time
        
        assert end > start
        assert total_time != 0 
        assert prop != 0

    if prop > 1:
        prop = 1

    return prop

def get_img_ind(img_path):
    filename = os.path.splitext(os.path.basename(img_path))[0]
    img_ind  = int(filename.split("RUN")[1])
    return img_ind 

def safe_open(path):
    try:
        img = np.array(Image.open(path))
    except UnidentifiedImageError:
        img = None 
    return img 

def prevent_empty_well_image(image_paths,img_ind,mask,thres=20):
    
    img = safe_open(image_paths[img_ind])

    if img is not None:
        mask = transform.resize(mask,img.shape,order=0)
        std = img[~mask].std()
    else:
        std = None

    if (img is None) or std < thres:

        while (img is None or std < thres) and img_ind > 0:
            img = safe_open(image_paths[img_ind])
            if img is None:
                img_ind -= 1 
            else:
                std = img[~mask].std()
                if std < thres:
                    img_ind -= 1 

        if ((std is None) or std < thres or (img is None)) and img_ind == 0:
            img_ind = None
 
    return img_ind

def get_focal_plane(path):
    focal_plane = path.split("/")[-2]
    focal_plane = int(focal_plane.replace("F",""))
    return focal_plane

def find_sharpest_focal_plan(path,img_ind):

    try:
        focal_plane_list = sorted(glob.glob(os.path.join(path,"F*/")),key=get_focal_plane)
    except ValueError:
        print(os.path.join(path,"/*/"),path)
        sys.exit(0)

    fold,well_id = path.split("/")[-2].split("-")
    
    sharpest_focal_plane = None
    highest_sharpness_value = 0
    for focal_plane in focal_plane_list:
        img_path = os.path.join(path,focal_plane,fold+"_WELL"+well_id+"_RUN"+str(img_ind)+".jpeg")
        if os.path.exists(img_path):
            img = Image.open(img_path)
            img = np.asarray(img, dtype=np.int32)
            gy, gx = np.gradient(img)
            gnorm = np.sqrt(gx**2 + gy**2)
            sharpness = np.average(gnorm)
        
            if sharpness > highest_sharpness_value:
                highest_sharpness_value = sharpness 
                sharpest_focal_plane = focal_plane
    
    sharpest_focal_plane = sharpest_focal_plane.split("/")[-2]

    return "F0",sharpest_focal_plane

def exctract_blast_img(path,dest_fold,img_ind,mask,empty_list,var_thres,frame_and_plane_dicts):
    
    #img_name = os.path.basename(paths[0])#.split("_RUN")[0]+".jpeg"
    #if not os.path.exists(dest_path):
    
    orig_img_ind = img_ind

    fold_name = path.split("/")[-2]

    if fold_name in frame_and_plane_dicts:

        dic = frame_and_plane_dicts[fold_name]

        img_ind = dic["frame_to_use"]

        if "focal_plane" in dic:
            focal_plane = dic["focal_plane"]
            #print("\t",path)
            sharpest_focal_plane = find_sharpest_focal_plan(path,img_ind)
        else:
            focal_plane,sharpest_focal_plane = find_sharpest_focal_plan(path,img_ind)

    else:
        focal_plane = "F0"
        paths = sorted(glob.glob(os.path.join(path,focal_plane,"*.*")),key=get_img_ind)
        img_ind = prevent_empty_well_image(paths,img_ind,mask,thres=var_thres)
        if not img_ind is None:
            focal_plane,sharpest_focal_plane = find_sharpest_focal_plan(path,img_ind)
        else:
            focal_plane,sharpest_focal_plane = "F0",None

    #if focal_plane != "F0":
    #    paths = sorted(glob.glob(os.path.join(path,focal_plane,"*.*")),key=get_img_ind)
    
    if img_ind is None:
        empty_list.append(path)
    else:
        #img_path = paths[img_ind]
        try:
            img_path = glob.glob(os.path.join(path,focal_plane,"*_RUN"+str(img_ind)+".jpeg"))[0]
        except IndexError:
            print(os.path.join(path,focal_plane,"*_RUN"+str(img_ind)+".jpeg"),orig_img_ind)
            sys.exit(0)

        img_name = os.path.basename(img_path)
        dest_path = os.path.join(dest_fold,img_name)
        if not os.path.exists(dest_path):
            os.makedirs(dest_fold,exist_ok=True)
            shutil.copyfile(img_path,dest_path)

    return empty_list,sharpest_focal_plane

def main():

    data_root = "/media/E144069X/DL4IVF/DL4IVF/"
    dest_dataset_root = "../data/dl4ivf_grade_dataset/"

    var_thres = 20
    h,w,r = 500,500,200
    x = np.arange(w)[np.newaxis]
    y = np.arange(h)[:,np.newaxis]
    center_x,center_y = w//2,h//2
    mask = (np.sqrt((x-center_x)**2+(y-center_y)**2) > r)
    
    #Make dataset
    #if not os.path.exists(dest_dataset_root+"/blast_and_grade_annot"):

    os.makedirs(dest_dataset_root,exist_ok=True)

    print("Reading csv")
    annot_root = "../data/"
    early_csv = pd.read_csv(os.path.join(annot_root,"export 2011-2016.csv"),sep=",",low_memory=False)
    late_csv = pd.read_csv(os.path.join(annot_root,"EXPORT 2017-2019.csv"),sep=";",low_memory=False)

    frame_and_plane_dict = make_frame_and_plane_dict(dest_dataset_root)

    print("Gathering folder paths")
    paths = glob.glob(os.path.join(data_root,"./*/"))
    nbVids = len(paths)

    grade_annot_csv = []
    grade_annot_list,blast_annot_list,no_annot_list = 0,0,0
    grade_diff_count = 0
    uncomplete_folders = []

    empty_list = []

    sharpest_focal_plane_list = []

    for i,path in enumerate(paths):

        if i %500==0:
            print(i,"/",len(paths))

        slideId,wellId = path.split("/")[-2].split("-")
        wellId = int(wellId)

        year = int(slideId.split(".")[0].replace("D",""))

        if year < 2017:
            csv = early_csv
        else:
            csv = late_csv

        bool_array = (csv["Slide ID"] ==  slideId) & (csv["Well"] ==  wellId)

        if bool_array.sum() > 0:

            rows = csv[bool_array]
            row = rows.iloc[0]

            te = row["TE - Value"]
            icm = row["ICM - Value"]
            is_blasto = str(row["tB"]) != "nan"
            is_expanded_blasto = str(row["tEB"]) != "nan"
            row_csv = [slideId+"-"+str(wellId),is_blasto,is_expanded_blasto,te,icm]
            grade_annot_csv.append(row_csv)

            if str(te) != "nan" and str(icm) != "nan":    
                dest_fold = os.path.join(dest_dataset_root,"blast_and_grade_annot/")
                
                te_time = str_to_float(row["TE - Time"])
                icm_time = str_to_float(row["ICM - Time"])

                time = (te_time + icm_time)/2
                #print("grade",te_time,icm_time)
                grade_annot_list += time

            elif is_blasto and is_expanded_blasto:
                dest_fold = os.path.join(dest_dataset_root,"blast_and_no_grade_annot/")
                time = str_to_float(row["tEB"])
                #print("blast",time)
                blast_annot_list += time

            #elif str(te) == "nan" and str(icm) == "nan" and (not is_blasto) and (not is_expanded_blasto):
            #    no_annot_list += 1
            #    dest_fold = os.path.join(dest_dataset_root,"no_blast_and_no_grade_annot/")
            #    time = -1
            #    print("noannot",time)
            #    no_annot_list += time

            else:
                time = None 

            if time is not None:
                img_nb = get_img_nb(path)
                if img_nb > 0:
                    img_ind = int(img_nb*get_prop(row,time)) - 1
                    img_ind = max(img_ind,0)
                    empty_list,sharpest_focal_plane = exctract_blast_img(path,dest_fold,img_ind,mask,empty_list,var_thres,frame_and_plane_dict)
                    sharpest_focal_plane_list.append(sharpest_focal_plane)
                else:
                    uncomplete_folders.append(path)

    np.savetxt("../data/uncomplete_folders.csv", uncomplete_folders, fmt='%s',delimiter=",")

    grade_annot_csv = np.array(grade_annot_csv)
    np.savetxt("../data/grade_annot.csv", grade_annot_csv, fmt='%s',delimiter=",",header="video,is_blasto,is_expanded_blasto,te,icm")

    empty_list = np.array(empty_list)
    np.savetxt("../data/empty_list.csv", empty_list, fmt='%s',delimiter=",")
 
    sharpest_focal_plane_list = np.array(sharpest_focal_plane_list)
    np.savetxt("../data/sharpest_focal_plane.csv", sharpest_focal_plane_list, fmt='%s',delimiter=",")

    #else:
    #    print("Dataset already exists")

    light_thres = 100
 
    #Fix contrast 
    folds = [dest_dataset_root+"/blast_and_grade_annot/",dest_dataset_root+"/blast_and_no_grade_annot/"]
    for j,fold in enumerate(folds):

        new_fold = fold[:-1]+"_fixed_contrast/"
        os.makedirs(new_fold,exist_ok=True)

        img_paths = sorted(glob.glob(fold+"/*.jpeg"))
        mean_list = []
        corrupted_list = []

        var_list = []
        embryo_hist_list= []
        no_embryo_hist_list = []

        if not os.path.exists(fold[:-1]+".csv"):
            img_names = ["names"]+[os.path.basename(path) for path in img_paths]
            csv_path = fold[:-1]+".csv"
            np.savetxt(csv_path,np.array(img_names), fmt='%s',delimiter=",")
            
        for i,path in enumerate(img_paths):  

            filename = os.path.basename(path)

            try:
                img = Image.open(path)
            except UnidentifiedImageError:
                corrupted_list.append(filename)
                continue

            lighthness = np.array(img).mean()

            if lighthness < light_thres:
                enhancer = ImageEnhance.Contrast(img)
                min_fact,max_fact = 1,1

                interp = (light_thres - lighthness)/light_thres
                factor = min_fact*(1-interp)+max_fact*interp
                img = enhancer.enhance(factor)

                enhancer = ImageEnhance.Brightness(img)
                min_fact,max_fact = 1,2.5
                factor = min_fact*(1-interp)+max_fact*interp
                img = enhancer.enhance(factor)
                
            mean_list.append(lighthness)

            if np.array(img).shape[0] != h:
                img = np.array(img)
                old_shape = img.shape
                img = transform.resize(img,(h,w),order=3)
                img = (255*img).astype("uint8")
                img = Image.fromarray(img)

            img.save(new_fold+"/"+filename)

            if i ==0:
                img = np.array(img)
                img[mask] = 0
                img = Image.fromarray(img)
                img.save(dest_dataset_root+f"masked_image{j}.png")

            img = np.array(img)
            var = img[~mask].std()
            var_list.append(var)

            if var < var_thres:
                no_embryo_hist_list.append(np.histogram(img,range=(0,255))[0])
                img = Image.fromarray(img)
                os.makedirs(dest_dataset_root+"/no_embryo/",exist_ok=True)
                img.save(dest_dataset_root+"/no_embryo/"+str(round(var))+"_"+filename)               
            else:
                embryo_hist_list.append(np.histogram(img,range=(0,255))[0])
                img = Image.fromarray(img)
                os.makedirs(dest_dataset_root+"/embryo/",exist_ok=True)
                img.save(dest_dataset_root+"/embryo/"+str(round(var))+"_"+filename)  

        
        no_embryo_hist_list = np.stack(no_embryo_hist_list,axis=0).mean(axis=0)
        embryo_hist_list = np.stack(embryo_hist_list,axis=0).mean(axis=0)

        plt.figure()
        plt.bar(np.arange(len(no_embryo_hist_list)),no_embryo_hist_list)
        plt.savefig(new_fold+"/no_emb_hist.png")
        plt.close()

        plt.figure()
        plt.bar(np.arange(len(embryo_hist_list)),embryo_hist_list)
        plt.savefig(new_fold+"/emb_hist.png")
        plt.close()        

        plt.figure()
        plt.hist(mean_list)
        plt.savefig(new_fold+"/hist.png")
        plt.close()

        plt.figure()
        plt.hist(var_list)
        plt.savefig(new_fold+"/var_hist.png")
        plt.close()       
        
    subprocess.run(["sudo","chown","-R","E144069X",dest_dataset_root])

if __name__ == "__main__":
    main()
    #dest_dataset_root = "../data/dl4ivf_grade_dataset/"
    #fix_annot_files(dest_dataset_root+"/blast_and_grade_annot/")
    #fix_annot_files(dest_dataset_root+"/blast_and_no_grade_annot/")

