
case $1 in
    "noneRed")
        python generate_representations.py -c model_emb10.config --model_id noneRed --stride_lay3 2 --stride_lay4 2
        python generate_representations.py -c model_emb10.config --model_id noneRed --stride_lay3 2 --stride_lay4 2 --transf black_patches
    ;;
    "none_mast")
        #python generate_representations.py -c model_emb10.config --model_id none_mast 
        #python generate_representations.py -c model_emb10.config --model_id none_mast --transf black_patches
        #python generate_representations.py -c model_emb10.config --model_id none_mast --transf black_patches_blur
        python generate_representations.py -c model_emb10.config --model_id none_mast --transf black_patches_size20
        python generate_representations.py -c model_emb10.config --model_id none_mast --transf black_patches_size60
        python generate_representations.py -c model_emb10.config --model_id none_mast --transf black_patches_blur_size20
        python generate_representations.py -c model_emb10.config --model_id none_mast --transf black_patches_blur_size60
    ;;
    "none_mast_img")
        python generate_representations.py -c model_emb10.config --model_id none_mast --transf img_size20
        python generate_representations.py -c model_emb10.config --model_id none_mast --transf img_size60
        python generate_representations.py -c model_emb10.config --model_id none_mast --transf img_blur_size20
        python generate_representations.py -c model_emb10.config --model_id none_mast --transf img_blur_size60
    ;;
    "none_mast_img_equal_masked_prop")
        python generate_representations.py -c model_emb10.config --model_id none_mast --transf black_patches_size20_nb90
        python generate_representations.py -c model_emb10.config --model_id none_mast --transf black_patches_size60_nb10
        python generate_representations.py -c model_emb10.config --model_id none_mast --transf img_blur_size20_nb90
        python generate_representations.py -c model_emb10.config --model_id none_mast --transf img_blur_size60_nb10
    ;;
    "spars_test")
        python generate_representations.py -c model_emb10.config --model_id none_mast --transf black_patches_nb10 --spars --class_map
    ;;
    "spars_test2")
        python generate_representations.py -c model_emb10.config --model_id none_mast --transf black_patches_nb120 --spars 
    ;;
    "none_mast_classmap")
        python generate_representations.py -c model_emb10.config --model_id none_mast --class_map --transf black_patches_size60
        python generate_representations.py -c model_emb10.config --model_id none_mast --class_map --transf black_patches_blur_size60
        python generate_representations.py -c model_emb10.config --model_id none_mast --class_map --transf img_size60
        python generate_representations.py -c model_emb10.config --model_id none_mast --class_map --transf img_blur_size60
    ;;
esac