
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
        python generate_representations.py -c model_emb10.config --model_id none_mast --transf black_patches_size4_nb10 --spars
        python generate_representations.py -c model_emb10.config --model_id none_mast --transf black_patches_size4_nb120 --spars 
    ;;
    "none_mast_classmap")
        python generate_representations.py -c model_emb10.config --model_id none_mast --class_map --transf black_patches_size60
        python generate_representations.py -c model_emb10.config --model_id none_mast --class_map --transf black_patches_blur_size60
        python generate_representations.py -c model_emb10.config --model_id none_mast --class_map --transf img_size60
        python generate_representations.py -c model_emb10.config --model_id none_mast --class_map --transf img_blur_size60
    ;;
    "noneRed_crohn")
        python generate_representations.py -c model_crohn2.config --model_id noneRed --resnet_bilinear False --big_images True  --stride_lay3 2 --stride_lay4 2 --transf saliency_metrics
        python generate_representations.py -c model_crohn2.config --model_id noneRed --resnet_bilinear False --big_images True  --stride_lay3 2 --stride_lay4 2 --transf identity
    ;;
    "noneRed_attMetrMask_crohn")
        python generate_representations.py -c model_crohn2.config --model_id 2_noneRed_attMetrMask --resnet_bilinear False --big_images True  --stride_lay3 2 --stride_lay4 2 --transf saliency_metrics
        python generate_representations.py -c model_crohn2.config --model_id 2_noneRed_attMetrMask --resnet_bilinear False --big_images True  --stride_lay3 2 --stride_lay4 2 --transf identity
    ;;
    "noneRed_0.5attMetrMask_FT_crohn")
        python generate_representations.py -c model_crohn2.config --model_id noneRed_0.5attMetrMask_FT --resnet_bilinear False --big_images True  --stride_lay3 2 --stride_lay4 2 --transf saliency_metrics
        python generate_representations.py -c model_crohn2.config --model_id noneRed_0.5attMetrMask_FT --resnet_bilinear False --big_images True  --stride_lay3 2 --stride_lay4 2 --transf identity
    ;;
    "noneRed_0.5attMetrMask_FT_simclr_crohn")
        python generate_representations.py -c model_crohn2.config --model_id noneRed_0.5attMetrMask_FT_simclr --resnet_bilinear False --big_images True  --stride_lay3 2 --stride_lay4 2 --transf saliency_metrics
        python generate_representations.py -c model_crohn2.config --model_id noneRed_0.5attMetrMask_FT_simclr --resnet_bilinear False --big_images True  --stride_lay3 2 --stride_lay4 2 --transf identity
    ;;
    "noneRed_0.5attMetrMask_FT_0.0016simclr")
        python generate_representations.py -c model_crohn2.config --model_id noneRed_0.5attMetrMask_FT_0.0016simclr --resnet_bilinear False --big_images True  --stride_lay3 2 --stride_lay4 2 --transf saliency_metrics
        python generate_representations.py -c model_crohn2.config --model_id noneRed_0.5attMetrMask_FT_0.0016simclr --resnet_bilinear False --big_images True  --stride_lay3 2 --stride_lay4 2 --transf identity
    ;;
    "cub_noneRed_focal2")
        for ind in 1 2 3
        do
            python generate_representations.py -c model_cub25.config --model_id noneRed_focal2 --resnet_bilinear False --big_images True  --stride_lay3 2 --stride_lay4 2 --transf saliency_metrics --layer $ind
            python generate_representations.py -c model_cub25.config --model_id noneRed_focal2 --resnet_bilinear False --big_images True  --stride_lay3 2 --stride_lay4 2 --transf identity --layer $ind
        done
    ;;
    "crohn_noneRed_focal2")
        for ind in 1 2 3
        do
            python generate_representations.py -c model_crohn25.config --model_id noneRed_focal2 --resnet_bilinear False --big_images True  --stride_lay3 2 --stride_lay4 2 --transf saliency_metrics --layer $ind
            python generate_representations.py -c model_crohn25.config --model_id noneRed_focal2 --resnet_bilinear False --big_images True  --stride_lay3 2 --stride_lay4 2 --transf identity --layer $ind
        done
    ;;
esac


