case $1 in
    "test")
        python trainVal.py -c model_emb10.config --optuna False --attention_metrics Del --model_id noneRed                               --att_metrics_post_hoc "gradcam_pp" --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True --att_metr_do_again True         --att_metr_img_bckgr "IB"
        python trainVal.py -c model_emb10.config --optuna False --attention_metrics Del --model_id noneRed                               --att_metrics_post_hoc "gradcam_pp" --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True --att_metr_do_again True         --att_metr_img_bckgr "black"
        python trainVal.py -c model_emb10.config --optuna False --attention_metrics Del --model_id noneRed                               --att_metrics_post_hoc "gradcam_pp" --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True --att_metr_do_again False         --att_metr_img_bckgr "gray"
        python trainVal.py -c model_emb10.config --optuna False --attention_metrics Del --model_id noneRed                               --att_metrics_post_hoc "gradcam_pp" --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True --att_metr_do_again False         --att_metr_img_bckgr "white"
        python trainVal.py -c model_emb10.config --optuna False --attention_metrics Del --model_id noneRed                               --att_metrics_post_hoc "gradcam_pp" --stride_lay3 2 --stride_lay4 2 --att_metrics_few_steps True --att_metr_do_again False         --att_metr_img_bckgr "blur"
        ;; 
    "*")
        echo "no such model"
    ;;
esac


