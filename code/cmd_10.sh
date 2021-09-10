
python3 processResults.py -c model_cub10.config --exp_id CUB10 --plot_points_image_dataset_grid --image_nb 100 \
													--model_ids clus_masterClusRed        \
													--plot_id intro --nrows 5 \
													--ind_to_keep 34 22 47

python3 processResults.py -c model_cub10.config --exp_id CUB10 --plot_points_image_dataset_grid --image_nb 100 \
													--model_ids clus_masterClusRed      \
													--plot_id grouping_algo  \
													--ind_to_keep 97 22 6 23 70 47 --nrows 3 --print_ind True

python3 processResults.py -c model_cub10.config --exp_id CUB10 --plot_points_image_dataset_grid --image_nb 12 \
													--model_ids clusRed clus_masterClusRed        \
													--plot_id strides_clus

python3 processResults.py -c model_cub10.config --exp_id CUB10 --plot_points_image_dataset_grid --image_nb 12 \
													--model_ids noneRed none_mast        \
													--plot_id strides_none \
													--ind_to_keep 1 3 9 10

################# Other datasets ##############################

python3 processResults.py -c model_cars10.config --exp_id CARS10 --plot_points_image_dataset_grid --image_nb 200 \
													--model_ids clus_mast \
													--plot_id grouping_algo_cars \
													--ind_to_keep  3 5 6 8 198 10 --nrows 3 --print_ind True

python3 processResults.py -c model_cars10.config --exp_id CARS10 --plot_points_image_dataset_grid --image_nb 12 \
													--model_ids clus_mast \
													--plot_id cars_intro \
													--ind_to_keep  3 5 8 --nrows 6

python3 processResults.py -c model_air10.config --exp_id AIR10 --plot_points_image_dataset_grid --image_nb 12 \
													--model_ids clus_mast   \
													--plot_id grouping_algo_air \
													--ind_to_keep 2 3 4 5 6 9 --nrows 3 --print_ind True

python3 processResults.py -c model_air10.config --exp_id AIR10 --plot_points_image_dataset_grid --image_nb 200 \
													--model_ids clus_mast   \
													--plot_id air_intro \
													--ind_to_keep 5 6 9 --nrows 3

python3 processResults.py -c model_emb10.config --exp_id EMB10 --plot_points_image_dataset_grid --image_nb 12 \
													--model_ids clus_mast        \
													--plot_id clus_mast_emb


################################### Part hierarch ###################################

python3 processResults.py -c model_cub10.config --exp_id CUB10 --plot_points_image_dataset_grid --image_nb 100 \
													--model_ids clus_masterClusRed      \
													--plot_id part_hier_birds  \
													--ind_to_keep 22 70 47 --nrows 3

python3 processResults.py -c model_air10.config --exp_id AIR10 --plot_points_image_dataset_grid --image_nb 12 \
													--model_ids clus_mast   \
													--plot_id part_hier_air \
													--ind_to_keep 2 6 9 --nrows 3

python3 processResults.py -c model_cars10.config --exp_id CARS10 --plot_points_image_dataset_grid --image_nb 200 \
													--model_ids clus_mast \
													--plot_id part_hier_cars \
													--ind_to_keep  3 5 198 --nrows 3 

################################ Vis comp ############################################""

python3 processResults.py -c model_cub10.config --exp_id CUB10 --plot_points_image_dataset_grid --image_nb 100 \
													--model_ids noneRed noneRed noneRed noneRed bilRed clus_masterClusRed      \
													--plot_id comp  \
													--ind_to_keep 47 29 84 --nrows 1 \
													--only_norm    False False False True False False \
													--gradcam 			True True True False False False \
													--gradcam_maps True False False False False False \
													--gradcam_pp   False False True False False False \
													--interp 				True True True True True False \
													--direct_ind    True True True False False False

python3 processResults.py -c model_cars10.config --exp_id CARS10 --plot_points_image_dataset_grid --image_nb 200 \
													--model_ids noneRed  noneRed noneRed noneRed bilRed clus_mast \
													--plot_id comp_cars \
													--ind_to_keep  135 39 84 --nrows 1 \
													--only_norm    False  False False True  False False \
													--gradcam 			True   True True  False False False \
													--gradcam_maps True   False False False False False \
													--gradcam_pp   False  False True  False False False \
													--interp 				True   True True  True  True  False \
													--direct_ind 		True  True True False False False

python3 processResults.py -c model_air10.config --exp_id AIR10 --plot_points_image_dataset_grid --image_nb 200 \
													--model_ids noneRed noneRed noneRed noneRed bilRed clus_mast   \
													--plot_id comp_air \
													--ind_to_keep 61 78 126 --nrows 1 \
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

##################### Vis compt with RISE #################################

python3 processResults.py -c model_cub10.config --exp_id CUB10 --plot_points_image_dataset_grid --image_nb 100 \
									--model_ids noneRed noneRed noneRed noneRed noneRed noneRed noneRed noneRed bilRed interbyparts modelprototree clus_masterClusRed \
									--plot_id comp_rise  \
									--ind_to_keep 47 29 84 --nrows 1 \
									--only_norm      False False True  False False False	False False False False False False \
									--rise 		     False False False True  False False	False False False False False False \
									--gradcam 	     True  True  False False True 	True	True  True  False False False False \
									--gradcam_maps   False False False False False True		False False False False False False \
									--gradcam_pp     False True  False False False False	False False False False False False \
									--score_map      False False False False True	False	False False False False False False \
									--vargrad        False False False False False	False	True  False False False False False \
									--smoothgrad_sq  False False False False False	False	False True  False False False False \
									--interp 	     True  True  True  True  True 	False	True  True  True  True  True False \
									--direct_ind     True  True  False True  True 	True	True  True  False False False False

python3 processResults.py -c model_air10.config --exp_id AIR10 --plot_points_image_dataset_grid --image_nb 200 \
													--model_ids noneRed noneRed noneRed noneRed noneRed noneRed bilRed clus_mast     \
													--plot_id comp_rise_air  \
													--ind_to_keep 61 78 126 --nrows 1 \
													--only_norm    False False False True  False False 	False False \
													--rise 		   False False False False True  False 	False False \
													--gradcam 	   True  True  True  False False True 	False False \
													--gradcam_maps True  False False False False False 	False False \
													--gradcam_pp   False False True  False False False 	False False \
													--score_map    False False False False False True	False False \
													--interp 	   True  True  True  True  True  True 	True  False \
													--direct_ind   True  True  True  False True  True 	False False

python3 processResults.py -c model_cars10.config --exp_id CARS10 --plot_points_image_dataset_grid --image_nb 200 \
													--model_ids noneRed noneRed noneRed noneRed noneRed noneRed bilRed clus_mast       \
													--plot_id comp_rise_cars  \
													--ind_to_keep 135 39 84 --nrows 1 \
													--only_norm    False False False True  False False 	False False \
													--rise 		   False False False False True  False 	False False \
													--gradcam 	   True  True  True  False False True 	False False \
													--gradcam_maps True  False False False False False 	False False \
													--gradcam_pp   False False True  False False False 	False False \
													--score_map    False False False False False True	False False \
													--interp 	   True  True  True  True  True  True 	True  False \
													--direct_ind   True  True  True  False True  True 	False False