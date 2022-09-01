case $1 in
    "Del")
        for bckgr in "black" "gray" "white" "blur" "IB"
        do
            python trainVal.py -c model_emb10.config --optuna False --attention_metrics Del --model_id clus_mast    --resnet_bilinear True --bil_cluster True                                                       --att_metrics_few_steps True --att_metr_do_again True     --att_metr_bckgr $bckgr  --att_metr_save_feat True --att_metr_add_first_feat True
            python trainVal.py -c model_emb10.config --optuna False --attention_metrics Del --model_id bilRed       --resnet_bilinear True                                          --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True --att_metr_do_again True     --att_metr_bckgr $bckgr  --att_metr_save_feat True --att_metr_add_first_feat True         
            python trainVal.py -c model_emb10.config --optuna False --attention_metrics Del --model_id abn          --abn True --abn_pretrained False --big_images False            --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True --att_metr_do_again True     --att_metr_bckgr $bckgr  --att_metr_save_feat True --att_metr_add_first_feat True
            python trainVal.py -c model_emb10.config --optuna False --attention_metrics Del --model_id interbyparts --inter_by_parts True                                           --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True --att_metr_do_again True     --att_metr_bckgr $bckgr  --att_metr_save_feat True --att_metr_add_first_feat True
            python trainVal.py -c model_emb10.config --optuna False --attention_metrics Del --model_id noneRed                                                                      --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True --att_metr_do_again True     --att_metr_bckgr $bckgr  --att_metr_save_feat True --att_metr_add_first_feat True
            python trainVal.py -c model_emb10.config --optuna False --attention_metrics Del --model_id noneRed                               --att_metrics_post_hoc "gradcam_pp"    --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True --att_metr_do_again True     --att_metr_bckgr $bckgr  --att_metr_save_feat True --att_metr_add_first_feat True
            python trainVal.py -c model_emb10.config --optuna False --attention_metrics Del --model_id noneRed                               --att_metrics_post_hoc "rise"          --stride_lay3 2 --stride_lay4 2                              --att_metr_do_again True     --att_metr_bckgr $bckgr  --att_metr_save_feat True --att_metr_add_first_feat True
            python trainVal.py -c model_emb10.config --optuna False --attention_metrics Del --model_id noneRed                               --att_metrics_post_hoc "score_map"     --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True --att_metr_do_again True     --att_metr_bckgr $bckgr  --att_metr_save_feat True --att_metr_add_first_feat True  
            python trainVal.py -c model_emb10.config --optuna False --attention_metrics Del --model_id noneRed                               --att_metrics_post_hoc "ablation_cam"  --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True --att_metr_do_again True     --att_metr_bckgr $bckgr  --att_metr_save_feat True --att_metr_add_first_feat True     
        done
        ;; 
    "Add")
        for bckgr in "black" "gray" "white" "blur" "IB"
        do
            python trainVal.py -c model_emb10.config --optuna False --attention_metrics Add --model_id clus_mast    --resnet_bilinear True --bil_cluster True                                                       --att_metrics_few_steps True --att_metr_do_again True     --att_metr_bckgr $bckgr  --att_metr_save_feat True --att_metr_add_first_feat True
            python trainVal.py -c model_emb10.config --optuna False --attention_metrics Add --model_id bilRed       --resnet_bilinear True                                          --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True --att_metr_do_again True     --att_metr_bckgr $bckgr  --att_metr_save_feat True --att_metr_add_first_feat True         
            python trainVal.py -c model_emb10.config --optuna False --attention_metrics Add --model_id abn          --abn True --abn_pretrained False --big_images False            --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True --att_metr_do_again True     --att_metr_bckgr $bckgr  --att_metr_save_feat True --att_metr_add_first_feat True
            python trainVal.py -c model_emb10.config --optuna False --attention_metrics Add --model_id interbyparts --inter_by_parts True                                           --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True --att_metr_do_again True     --att_metr_bckgr $bckgr  --att_metr_save_feat True --att_metr_add_first_feat True
            python trainVal.py -c model_emb10.config --optuna False --attention_metrics Add --model_id noneRed                                                                      --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True --att_metr_do_again True     --att_metr_bckgr $bckgr  --att_metr_save_feat True --att_metr_add_first_feat True
            python trainVal.py -c model_emb10.config --optuna False --attention_metrics Add --model_id noneRed                               --att_metrics_post_hoc "gradcam_pp"    --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True --att_metr_do_again True     --att_metr_bckgr $bckgr  --att_metr_save_feat True --att_metr_add_first_feat True
            python trainVal.py -c model_emb10.config --optuna False --attention_metrics Add --model_id noneRed                               --att_metrics_post_hoc "rise"          --stride_lay3 2 --stride_lay4 2                              --att_metr_do_again True     --att_metr_bckgr $bckgr  --att_metr_save_feat True --att_metr_add_first_feat True
            python trainVal.py -c model_emb10.config --optuna False --attention_metrics Add --model_id noneRed                               --att_metrics_post_hoc "score_map"     --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True --att_metr_do_again True     --att_metr_bckgr $bckgr  --att_metr_save_feat True --att_metr_add_first_feat True  
            python trainVal.py -c model_emb10.config --optuna False --attention_metrics Add --model_id noneRed                               --att_metrics_post_hoc "ablation_cam"  --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True --att_metr_do_again True     --att_metr_bckgr $bckgr  --att_metr_save_feat True --att_metr_add_first_feat True     
        done
        ;;
    "Lift")
        for bckgr in "black" "gray" "white" "blur" "IB"
        do
            python trainVal.py -c model_emb10.config --optuna False --attention_metrics Lift --model_id clus_mast    --resnet_bilinear True --bil_cluster True                                                       --att_metrics_few_steps True --att_metr_do_again False     --att_metr_bckgr $bckgr --att_metr_save_feat True 
            python trainVal.py -c model_emb10.config --optuna False --attention_metrics Lift --model_id bilRed       --resnet_bilinear True                                          --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True --att_metr_do_again False     --att_metr_bckgr $bckgr --att_metr_save_feat True          
            python trainVal.py -c model_emb10.config --optuna False --attention_metrics Lift --model_id abn          --abn True --abn_pretrained False --big_images False            --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True --att_metr_do_again False     --att_metr_bckgr $bckgr --att_metr_save_feat True 
            python trainVal.py -c model_emb10.config --optuna False --attention_metrics Lift --model_id interbyparts --inter_by_parts True                                           --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True --att_metr_do_again False     --att_metr_bckgr $bckgr --att_metr_save_feat True 
            python trainVal.py -c model_emb10.config --optuna False --attention_metrics Lift --model_id noneRed                                                                      --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True --att_metr_do_again False     --att_metr_bckgr $bckgr --att_metr_save_feat True 
            python trainVal.py -c model_emb10.config --optuna False --attention_metrics Lift --model_id noneRed                               --att_metrics_post_hoc "gradcam_pp"    --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True --att_metr_do_again False     --att_metr_bckgr $bckgr --att_metr_save_feat True 
            python trainVal.py -c model_emb10.config --optuna False --attention_metrics Lift --model_id noneRed                               --att_metrics_post_hoc "rise"          --stride_lay3 2 --stride_lay4 2                              --att_metr_do_again False     --att_metr_bckgr $bckgr --att_metr_save_feat True 
            python trainVal.py -c model_emb10.config --optuna False --attention_metrics Lift --model_id noneRed                               --att_metrics_post_hoc "score_map"     --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True --att_metr_do_again False     --att_metr_bckgr $bckgr --att_metr_save_feat True   
            python trainVal.py -c model_emb10.config --optuna False --attention_metrics Lift --model_id noneRed                               --att_metrics_post_hoc "ablation_cam"  --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True --att_metr_do_again False     --att_metr_bckgr $bckgr --att_metr_save_feat True      
        done
        ;;
    "*")
        echo "no such model"
    ;;
esac


