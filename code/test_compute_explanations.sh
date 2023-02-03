python compute_explanations.py -c model_cub10.config --model_id noneRed  --inds 4816 4835 753  --exp_id CUB10   --stride_lay3 2 --stride_lay4 2 --att_metrics_post_hoc gradcam

#python compute_explanations.py -c model_cub10.config --model_id noneRed  --inds 4816 4835 753  --exp_id CUB10   --stride_lay3 2 --stride_lay4 2 --att_metrics_post_hoc gradcam_pp

#python compute_explanations.py -c model_cub10.config --model_id noneRed  --inds 4816 4835 753  --exp_id CUB10   --stride_lay3 2 --stride_lay4 2 --att_metrics_post_hoc score_cam

#python compute_explanations.py -c model_cub10.config --model_id noneRed  --inds 4816 4835 753  --exp_id CUB10   --stride_lay3 2 --stride_lay4 2 --att_metrics_post_hoc ablation_cam