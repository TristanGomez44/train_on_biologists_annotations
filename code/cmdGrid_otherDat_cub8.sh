model=optuna-avg

python3 processResults.py -c model.config --exp_id CARS8 --plot_points_image_dataset_grid --image_nb 12 \
													--model_ids	  				 $model      \
													--reduction_fact_list  4           \
													--inverse_xy           False  \
													--use_dropped_list     False  \
													--force_feat 					 False  \
													--full_att_map         True   \
													--use_threshold 			 False  \
													--maps_inds            -1     \
													--receptive_field			 False  \
													--agregate_multi_att   True   \
													--pond_by_norm				 True   \
													--mode test  --plot_id cars  \
													--dataset_test cars_test \
													--nrows 4

python3 processResults.py -c model.config --exp_id AIR8 --plot_points_image_dataset_grid --image_nb 12 \
													--model_ids	  				 $model      \
													--reduction_fact_list  4           \
													--inverse_xy           False   \
													--use_dropped_list     False   \
													--force_feat 					 False   \
													--full_att_map         True    \
													--use_threshold 			 False   \
													--maps_inds            -1       \
													--receptive_field			 False   \
													--agregate_multi_att   True     \
													--pond_by_norm				 True    \
													--mode test  --plot_id air  \
													--dataset_test aircraft_test \
													--nrows 4

python3 processResults.py -c model.config --exp_id DOGS8 --plot_points_image_dataset_grid --image_nb 12 \
													--model_ids	  				 $model      \
													--reduction_fact_list  4          \
													--inverse_xy           False   \
													--use_dropped_list     False   \
													--force_feat 					 False   \
													--full_att_map         True     \
													--use_threshold 			 False  \
													--maps_inds            -1         \
													--receptive_field			 False   \
													--agregate_multi_att   True   \
													--pond_by_norm				 True     \
													--mode test  --plot_id dogs  \
													--dataset_test dogs_test \
													--nrows 4
