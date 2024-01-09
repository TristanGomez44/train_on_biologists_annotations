import os 
import glob

import pandas as pd
from PIL import Image,ImageDraw,ImageFont

from args import ArgReader
from enums import Tasks 

def conf_to_label(confidence,low_thres,high_thres):
    if confidence < low_thres:
        return "faible"
    elif confidence < high_thres:
        return "moyenne"
    else:
        return "forte"

def get_annot_str(te,icm,exp,te_conf,icm_conf,exp_conf,low_conf_thres,high_conf_thres):
        annot_str = ""

        for label,annot,conf in zip(["TE","ICM","EXP"],[te,icm,exp],[te_conf,icm_conf,exp_conf]):
            
            conf = conf_to_label(conf,low_conf_thres,high_conf_thres)
            
            if annot != "Null":
                annot_str += f"{label} : {annot} (confiance {conf}) "
            else:
                annot_str += f"{label} : N/A"

            annot_str += "\n"

        return annot_str

def add_annot(img,te,icm,exp,te_conf,icm_conf,exp_conf,low_conf_thres,high_conf_thres,fontsize=25,margin=10,padd_size=50):
    
        # Create a new image with the same width and extra height for the banner
        new_width = img.width + padd_size
        new_height = img.height + padd_size  # You can adjust the banner height as needed
        new_img = Image.new('RGB', (new_width, new_height), color='black')

        # Paste the original image onto the new image
        new_img.paste(img, (padd_size//2, 0))

        # Create a drawing object
        draw = ImageDraw.Draw(new_img)

        # Choose a font (you can specify the font file and size)
        font = ImageFont.truetype("arial.ttf", fontsize)

        # Specify the position to place the text (in this case, at the bottom center)
        width, height = img.size
        
        banner_text = get_annot_str(te,icm,exp,te_conf,icm_conf,exp_conf,low_conf_thres,high_conf_thres)
        
        position = (margin,height-fontsize-margin)
        #textwidth, textheight = draw.textbbox((),banner_text, font=font)
        #text_width, text_height = draw.textsize(banner_text, font)
    
        # Specify the text color
        text_color = (255,255,255)

        # Add text to the image
        draw.text(position, banner_text, font=font, fill=text_color)

        return new_img

def main(argv=None):

    #Getting arguments from config file and command row
    #Building the arg reader
    argreader = ArgReader(argv)

    argreader.parser.add_argument('--imgs_folder',type=str)
    argreader.parser.add_argument('--aggr_annot_csv',type=str)
    argreader.parser.add_argument('--conf_csv',type=str)
    argreader.parser.add_argument('--output_dir',type=str)    
    argreader.parser.add_argument('--low_conf_thres',type=int,default=0.5)     
    argreader.parser.add_argument('--high_conf_thres',type=int,default=0.95)     

    #Reading the comand row arg
    argreader.getRemainingArgs()

    #Getting the args from command row and config file
    args = argreader.args
    
    csv = pd.read_csv(args.aggr_annot_csv,delimiter=" ")
    conf_csv = pd.read_csv(args.conf_csv,delimiter=" ")

    for task in Tasks:
        csv[task.value+"_conf"] = conf_csv[task.value]

    os.makedirs(args.output_dir,exist_ok=True)

    for _, (image_name,te,icm,exp,te_conf,icm_conf,exp_conf) in csv.iterrows():
        
        image_suff = image_name.replace("F0","")

        img_paths = glob.glob(os.path.join(args.imgs_folder,"*"+image_suff))
        
        for path in img_paths:
            print(path)
            image = Image.open(path)

            image_with_annot = add_annot(image,te,icm,exp,te_conf,icm_conf,exp_conf,args.low_conf_thres,args.high_conf_thres)

            filename = os.path.basename(path)
            image_with_annot.save(os.path.join(args.output_dir,filename))

if __name__ == "__main__":
    main()
