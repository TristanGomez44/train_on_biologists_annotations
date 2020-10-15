
python3 processResults.py -c model.config --exp_id CUB8 --plot_points_image_dataset_grid --image_nb 12 \
													--model_ids	 opt-bil-avg optuna-avg         \
													--reduction_fact_list 4     4         \
													--inverse_xy          False False   \
													--use_dropped_list    False False   \
													--force_feat 					False False   \
													--full_att_map        True  True    \
													--use_threshold 			False False   \
													--maps_inds           -1     -1       \
													--receptive_field			 False   \
													--agregate_multi_att   False   \
													--pond_by_norm				 True    \
													--mode test  --plot_id bil_vs_clus \
													--nrows 4

python3 processResults.py -c model.config --exp_id CUB5 --plot_points_image_dataset_grid --image_nb 15 \
													--model_ids	  				 optuna-avg      \
													--reduction_fact_list  4      \
													--inverse_xy           False  \
													--use_dropped_list     False  \
													--force_feat 					 False  \
													--full_att_map         True   \
													--use_threshold 			 False  \
													--maps_inds            -1      \
													--receptive_field			 False  \
													--agregate_multi_att   False   \
													--pond_by_norm				 True   \
													--mode test  --plot_id clus_intro  \
													--nrows 5

python3 processResults.py -c model.config --exp_id CUB8 --plot_points_image_dataset_grid --image_nb 15 \
														--model_ids	  none  \
														--reduction_fact_list   4     \
														--inverse_xy            False \
														--use_dropped_list      False \
														--force_feat 					  False \
														--full_att_map          True  \
														--use_threshold 			  False \
														--maps_inds             -1    \
														--cluster_attention			True	\
														--luminosity False \
														--nrows 5 \
														--mode test  --plot_id umap

python3 processResults.py -c model.config --exp_id CUB5 --plot_points_image_dataset_grid --image_nb 12 \
														--model_ids	  					noneRed  none \
														--reduction_fact_list   4     4       \
														--inverse_xy            False False    \
														--use_dropped_list      False False    \
														--force_feat 					  False False    \
														--full_att_map          True  True     \
														--use_threshold 			  False False    \
														--maps_inds             -1    -1       \
														--luminosity False \
														--nrows 4 \
														--mode test  --plot_id stride
