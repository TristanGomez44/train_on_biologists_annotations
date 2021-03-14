
python3 processResults.py -c model_cub10.config --exp_id CUB10 --plot_points_image_dataset_grid --image_nb 100 \
													--model_ids clus_masterClusRed        \
													--plot_id intro --nrows 5 \
													--ind_to_keep 34 22 47

python3 processResults.py -c model_cub10.config --exp_id CUB10 --plot_points_image_dataset_grid --image_nb 100 \
													--model_ids clus_masterClusRed      \
													--plot_id grouping_algo  \
													--ind_to_keep 97 22 6 82 70 47 --nrows 3

python3 processResults.py -c model_cub10.config --exp_id CUB10 --plot_points_image_dataset_grid --image_nb 12 \
													--model_ids clusRed clus_masterClusRed        \
													--plot_id strides_clus

python3 processResults.py -c model_cub10.config --exp_id CUB10 --plot_points_image_dataset_grid --image_nb 12 \
													--model_ids noneRed none_mast        \
													--plot_id strides_none

################# Other datasets ##############################

python3 processResults.py -c model_cars10.config --exp_id CARS10 --plot_points_image_dataset_grid --image_nb 12 \
													--model_ids clus_mast \
													--plot_id grouping_algo_cars \
													--ind_to_keep  3 5 6 8 9 10 --nrows 3

python3 processResults.py -c model_cars10.config --exp_id CARS10 --plot_points_image_dataset_grid --image_nb 12 \
													--model_ids clus_mast \
													--plot_id cars_intro \
													--ind_to_keep  3 5 8 --nrows 6

python3 processResults.py -c model_air10.config --exp_id AIR10 --plot_points_image_dataset_grid --image_nb 12 \
													--model_ids clus_mast   \
													--plot_id grouping_algo_air \
													--ind_to_keep 2 3 4 5 6 9 --nrows 3

python3 processResults.py -c model_air10.config --exp_id AIR10 --plot_points_image_dataset_grid --image_nb 12 \
													--model_ids clus_mast   \
													--plot_id air_intro \
													--ind_to_keep 5 6 9 --nrows 3

python3 processResults.py -c model_emb10.config --exp_id EMB10 --plot_points_image_dataset_grid --image_nb 12 \
													--model_ids clus_mast        \
													--plot_id clus_mast_emb


#python3 processResults.py -c model_cub10.config --exp_id CUB10 --plot_points_image_dataset_grid --image_nb 100 \
#													--model_ids clus_masterClusRed  bil_mast      \
#													--plot_id grouping_algo_diff \
#													--ind_to_keep 97 34 22 6 42 82 78 70 35 47 88 75

#python3 processResults.py -c model_cars10.config --exp_id CARS10 --plot_points_image_dataset_grid --image_nb 200 \
#													--model_ids clus_mast bil_mast \
#													--plot_id grouping_algo_cars \
#													--ind_to_keep 2 5 25 33 55 73 85 105 116 161 199 197

#python3 processResults.py -c model_air10.config --exp_id AIR10 --plot_points_image_dataset_grid --image_nb 120 \
#													--model_ids clus_mast bil_mast  \
#													--plot_id grouping_algo_air_diff \
#													--ind_to_keep 8 19 55 59 35 36 40 91 60 100 102 112


################################ Vis comp ############################################""

python3 processResults.py -c model_cub10.config --exp_id CUB10 --plot_points_image_dataset_grid --image_nb 100 \
													--model_ids noneRed noneRed noneRed noneRed bilRed clus_masterClusRed      \
													--plot_id comp  \
													--ind_to_keep 22 82 70 --nrows 1 \
													--only_norm    False False False True False False \
													--gradcam 			True True True False False False \
													--gradcam_maps True False False False False False \
													--gradcam_pp   False False True False False False \
													--interp 				True True True True True False \
													--direct_ind    True True True False False False

python3 processResults.py -c model_cars10.config --exp_id CARS10 --plot_points_image_dataset_grid --image_nb 12 \
													--model_ids noneRed  noneRed noneRed noneRed bilRed clus_mast \
													--plot_id comp_cars \
													--ind_to_keep  3 5 9 --nrows 1 \
													--only_norm    False  False False True  False False \
													--gradcam 			True   True True  False False False \
													--gradcam_maps True   False False False False False \
													--gradcam_pp   False  False True  False False False \
													--interp 				True   True True  True  True  False \
													--direct_ind 		True  True True False False False

python3 processResults.py -c model_air10.config --exp_id AIR10 --plot_points_image_dataset_grid --image_nb 12 \
													--model_ids noneRed noneRed noneRed noneRed bilRed clus_mast   \
													--plot_id comp_air \
													--ind_to_keep 5 6 9 --nrows 1 \
													--only_norm    False False False True False False \
													--gradcam 			True True True False False False \
													--gradcam_maps True  False  False False False False \
													--gradcam_pp   False False True False False False \
													--interp 				True True True True True False \
													--direct_ind    True True True False False False

python3 processResults.py -c model_cub10.config --exp_id CUB10 --plot_points_image_dataset_grid --image_nb 100 \
													--model_ids noneRed noneRed noneRed noneRed bilRed clus_masterClusRed      \
													--plot_id teaser  \
													--ind_to_keep 82 --nrows 1 \
													--only_norm    False False False True False False \
													--gradcam 			True True True False False False \
													--gradcam_maps True False False False False False \
													--gradcam_pp   False False True False False False \
													--interp 				True True True True True False \
													--direct_ind    True True True False False False

python3 processResults.py -c model_cars10.config --exp_id CARS10 --plot_points_image_dataset_grid --image_nb 12 \
													--model_ids noneRed  noneRed noneRed noneRed bilRed clus_mast \
													--plot_id teaser_cars \
													--ind_to_keep  9 --nrows 1 \
													--only_norm    False  False False True  False False \
													--gradcam 			True   True True  False False False \
													--gradcam_maps True   False False False False False \
													--gradcam_pp   False  False True  False False False \
													--interp 				True   True True  True  True  False \
													--direct_ind 		True  True True False False False
