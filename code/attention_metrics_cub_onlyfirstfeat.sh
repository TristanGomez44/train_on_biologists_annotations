#python trainVal.py -c model_cub10.config --optuna False --attention_metrics Del --model_id noneRed                               --att_metrics_post_hoc "gradcam_pp"    --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True --att_metr_do_again True     --att_metr_bckgr black  --att_metr_add_first_feat True 

python trainVal.py -c model_cub10.config --optuna False --attention_metrics Add --model_id noneRed                               --att_metrics_post_hoc "gradcam_pp"    --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True --att_metr_do_again True     --att_metr_bckgr blur  --att_metr_add_first_feat True 