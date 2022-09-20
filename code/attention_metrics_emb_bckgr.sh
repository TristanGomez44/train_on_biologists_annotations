if [ "$1" != "all" ];
then
    declare -a metr_list=("Del" "Add" "Lift")
else
    declare -a metr_list=($2)
fi

if [ "$2" != "all" ];
then
    declare -a bckgr_list=("highpass" "lowpass" "edge" "black" "gray" "white" "blur" "IB")
else
    declare -a bckgr_list=($2)
fi

for metric in "${metr_list[@]}"
do
    for bckgr in "${bckgr_list[@]}"
    do
        echo BCKGR $bckgr
        python trainVal.py -c model_emb10.config --optuna False --attention_metrics $metric --model_id clus_mast    --resnet_bilinear True --bil_cluster True                                                       --att_metrics_few_steps True --att_metr_do_again False     --att_metr_bckgr $bckgr  --att_metr_save_feat True 
        python trainVal.py -c model_emb10.config --optuna False --attention_metrics $metric --model_id bilRed       --resnet_bilinear True                                          --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True --att_metr_do_again False     --att_metr_bckgr $bckgr  --att_metr_save_feat True          
        python trainVal.py -c model_emb10.config --optuna False --attention_metrics $metric --model_id abn          --abn True --abn_pretrained False --big_images False            --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True --att_metr_do_again False     --att_metr_bckgr $bckgr  --att_metr_save_feat True 
        python trainVal.py -c model_emb10.config --optuna False --attention_metrics $metric --model_id interbyparts --inter_by_parts True                                           --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True --att_metr_do_again False     --att_metr_bckgr $bckgr  --att_metr_save_feat True 
        python trainVal.py -c model_emb10.config --optuna False --attention_metrics $metric --model_id noneRed                                                                      --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True --att_metr_do_again False     --att_metr_bckgr $bckgr  --att_metr_save_feat True 
        python trainVal.py -c model_emb10.config --optuna False --attention_metrics $metric --model_id noneRed                               --att_metrics_post_hoc "gradcam_pp"    --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True --att_metr_do_again False     --att_metr_bckgr $bckgr  --att_metr_save_feat True 
        python trainVal.py -c model_emb10.config --optuna False --attention_metrics $metric --model_id noneRed                               --att_metrics_post_hoc "rise"          --stride_lay3 2 --stride_lay4 2                              --att_metr_do_again False     --att_metr_bckgr $bckgr  --att_metr_save_feat True 
        python trainVal.py -c model_emb10.config --optuna False --attention_metrics $metric --model_id noneRed                               --att_metrics_post_hoc "score_map"     --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True --att_metr_do_again False     --att_metr_bckgr $bckgr  --att_metr_save_feat True   
        python trainVal.py -c model_emb10.config --optuna False --attention_metrics $metric --model_id noneRed                               --att_metrics_post_hoc "ablation_cam"  --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True --att_metr_do_again False     --att_metr_bckgr $bckgr  --att_metr_save_feat True      
    done
done 
