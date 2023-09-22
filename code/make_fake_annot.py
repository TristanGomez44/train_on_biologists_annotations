import os
import random
import csv
import argparse

from enums import annot_enum_dic,Tasks

def main():

    # Create an argument parser
    parser = argparse.ArgumentParser(description="Generate random annotations for the DL4IVF blastocyst dataset.")
    parser.add_argument("--image_folder", help="Path to the DL4IVF dataset",default="../data/dl4ivf_blastocysts/")
    parser.add_argument("--annot_nb", help="Nb of annotation per image",default=5)
    parser.add_argument("--agreement_rate", help="Probability of obtaining the most common annotations.",default=0.9)
    parser.add_argument("--fake_annot_path", help="Path to the resulting fake annotation csv")

    # Parse the command-line arguments
    args = parser.parse_args()

    image_folder = args.image_folder
        
    if args.fake_annot_path is None:

        if image_folder.endswith("/"):
            image_folder = image_folder[:-1]

        assert image_folder.split("/")[-1] == "Images","Set the --fake_annot_path or set the --image_folder to the 'Image' folder of the dl4ivf dataset"
        fake_annot_path = os.path.dirname(image_folder)

    else:
        fake_annot_path = args.fake_annot_path
        
    image_files = [f for f in os.listdir(image_folder) if f.startswith("F0") and f.endswith(('.jpeg'))]
    annotations = []

    all_annot_dic = {task:list(annot_enum_dic[task]) for task in Tasks}
        
    for image_file in image_files:
         
        weights_dic = {}
        for task in Tasks:

            all_annot = all_annot_dic[task]
            most_common_annot = random.choice(all_annot)
            most_common_annot_prob = args.agreement_rate
            other_annot_prob = (1 - args.agreement_rate)/len(all_annot)
            weights = []

            for annot in all_annot:
                if annot == most_common_annot:
                    weights.append(most_common_annot_prob)
                else:
                    weights.append(other_annot_prob)

            weights_dic[task] = weights
            
        for _ in range(args.annot_nb):
            annot_dic = {'image_name':image_file}
            for task in Tasks:
                annot_dic[task.value] = random.choices(all_annot_dic[task],weights=weights_dic[task])[0].value

            annotations.append(annot_dic)

    csv_file = os.path.join(fake_annot_path,"fake_annotations.csv")

    # Write annotations to a CSV file
    with open(csv_file, 'w', newline='') as csvfile:
        fieldnames = ['image_name'] + [task.value for task in Tasks]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write the header row
        writer.writeheader()
        
        # Write image annotations
        for annotation in annotations:
            writer.writerow(annotation)

if __name__ == "__main__":
    main()