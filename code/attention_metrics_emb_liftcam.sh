case $1 in
    "all")
        python trainVal.py -c model_emb10.config --optuna False --attention_metrics Lift --model_id clus_mast       --resnet_bilinear True --bil_cluster True --att_metrics_few_steps True --att_metr_do_again False
        python trainVal.py -c model_emb10.config --optuna False --attention_metrics Lift --model_id noneRed                                                                  --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True  --att_metr_do_again False
        python trainVal.py -c model_emb10.config --optuna False --attention_metrics Lift --model_id bilRed             --resnet_bilinear True                                 --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True --att_metr_do_again False
        python trainVal.py -c model_emb10.config --optuna False --attention_metrics Lift --model_id abn          --abn True --abn_pretrained False --big_images False         --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True   --att_metr_do_again False
        python trainVal.py -c model_emb10.config --optuna False --attention_metrics Lift --model_id interbyparts --inter_by_parts True                                        --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True --att_metr_do_again False
        python trainVal.py -c model_emb10.config --optuna False --attention_metrics Lift --model_id noneRed                               --att_metrics_post_hoc "gradcam_pp" --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True --att_metr_do_again False
        python trainVal.py -c model_emb10.config --optuna False --attention_metrics Lift --model_id noneRed                               --att_metrics_post_hoc "rise"       --stride_lay3 2 --stride_lay4 2  --att_metr_do_again False
        python trainVal.py -c model_emb10.config --optuna False --attention_metrics Lift --model_id noneRed                               --att_metrics_post_hoc "score_map"  --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True  --att_metr_do_again False  
        python trainVal.py -c model_emb10.config --optuna False --attention_metrics Lift --model_id noneRed                               --att_metrics_post_hoc "xgradcam"  --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True  --att_metr_do_again False  
        python trainVal.py -c model_emb10.config --optuna False --attention_metrics Lift --model_id noneRed                               --att_metrics_post_hoc "ablation_cam"  --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True  --att_metr_do_again False  
    ;;
    "all_imgbckgr")
        python trainVal.py -c model_emb10.config --optuna False --attention_metrics Lift --model_id clus_mast       --resnet_bilinear True --bil_cluster True --att_metrics_few_steps True --att_metr_do_again False                                                      --att_metr_img_bckgr True
        python trainVal.py -c model_emb10.config --optuna False --attention_metrics Lift --model_id noneRed                                                                  --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True  --att_metr_do_again False      --att_metr_img_bckgr True    
        python trainVal.py -c model_emb10.config --optuna False --attention_metrics Lift --model_id bilRed             --resnet_bilinear True                                 --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True --att_metr_do_again False      --att_metr_img_bckgr True
        python trainVal.py -c model_emb10.config --optuna False --attention_metrics Lift --model_id abn          --abn True --abn_pretrained False --big_images False         --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True   --att_metr_do_again False    --att_metr_img_bckgr True
        python trainVal.py -c model_emb10.config --optuna False --attention_metrics Lift --model_id interbyparts --inter_by_parts True                                        --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True --att_metr_do_again False      --att_metr_img_bckgr True
        python trainVal.py -c model_emb10.config --optuna False --attention_metrics Lift --model_id noneRed                               --att_metrics_post_hoc "gradcam_pp" --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True --att_metr_do_again False      --att_metr_img_bckgr True
        python trainVal.py -c model_emb10.config --optuna False --attention_metrics Lift --model_id noneRed                               --att_metrics_post_hoc "rise"       --stride_lay3 2 --stride_lay4 2  --att_metr_do_again False                                  --att_metr_img_bckgr True
        python trainVal.py -c model_emb10.config --optuna False --attention_metrics Lift --model_id noneRed                               --att_metrics_post_hoc "score_map"  --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True  --att_metr_do_again False     --att_metr_img_bckgr True  
        python trainVal.py -c model_emb10.config --optuna False --attention_metrics Lift --model_id noneRed                               --att_metrics_post_hoc "xgradcam"  --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True  --att_metr_do_again False      --att_metr_img_bckgr True  
        python trainVal.py -c model_emb10.config --optuna False --attention_metrics Lift --model_id noneRed                               --att_metrics_post_hoc "ablation_cam"  --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True  --att_metr_do_again False  --att_metr_img_bckgr True 
    ;;
    "*")
        echo "no such model"
    ;;
esac


