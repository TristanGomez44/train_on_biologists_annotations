import os,argparse 

def main():

    #Parse arguments
    parser = argparse.ArgumentParser()

    #The path to the .tex file to convert to pdf 
    parser.add_argument("--text_path", type=str, help="The path to the .text file.")

    #Collect arguments

    args = parser.parse_args()

    #Get the path to the .tex file
    tex_path = args.text_path

    #Load tex file
    with open(tex_path, 'r') as f:
        tex = f.read()

    if not tex.startswith("\documentclass{standalone}"):
        latex_header = "\documentclass{standalone} \n \\begin{document}"
        tex = latex_header + tex
        latex_footer = "\end{document}"
        tex = tex + latex_footer
        with open(tex_path, 'w') as f:
            f.write(tex)
    
    #Convert tex to pdf and save it in the same folder
    os.system(f"pdflatex -output-directory {os.path.dirname(tex_path)} {tex_path}")

    #Remove aux files
    os.system(f"rm {tex_path[:-4]}.aux")
    os.system(f"rm {tex_path[:-4]}.log")
    os.system(f"rm {tex_path[:-4]}.out")

if __name__ == "__main__":
    main()