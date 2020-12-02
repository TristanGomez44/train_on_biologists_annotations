case $1 in
  "clusEarly")
    python trainVal.py -c model_cub9.config --model_id clusEarl   --bil_cluster_early True
    ;;
  "clusEarlyExp")
    python trainVal.py -c model_cub9.config --model_id clusEarlExp   --bil_cluster_early True --bil_clu_earl_exp True
    ;;
  "clusEarlyRed")
    python trainVal.py -c model_cub9.config --model_id clusEarlRed   --bil_cluster_early True --stride_lay3 2 --stride_lay4 2
    ;;
  "clusEarlyExpRed")
    python trainVal.py -c model_cub9.config --model_id clusEarlExpRed   --bil_cluster_early True --bil_clu_earl_exp True --stride_lay3 2 --stride_lay4 2
    ;;
  "clusAt1_GlobVec")
    python trainVal.py -c model_cub9.config --model_id clusAt1_GlobVec  --resnet_bilinear True --bil_cluster True  --bil_cluster_lay_ind 2 --bil_clu_glob_vec True --stride_lay3 2 --stride_lay4 2
    ;;
  "clusAt1_GlobRepVec")
    python trainVal.py -c model_cub9.config --model_id clusAt1_GlobRepVec  --resnet_bilinear True --bil_cluster True  --bil_cluster_lay_ind 2 --bil_clu_glob_rep_vec True --stride_lay3 2 --stride_lay4 2
    ;;
  "clusAt1_GlobRefRepVec")
    python trainVal.py -c model_cub9.config --model_id clusAt1_GlobRefRepVec  --resnet_bilinear True --bil_cluster True  --bil_cluster_lay_ind 2 --bil_clu_glob_rep_vec True --bil_cluster_glob_norefine False --apply_softmax_on_sim_glob True --stride_lay3 2 --stride_lay4 2
    ;;
  "clusAt1_GlobCorVec")
    python trainVal.py -c model_cub9.config --model_id clusAt1_GlobCorVec  --resnet_bilinear True --bil_cluster True  --bil_cluster_lay_ind 2 --bil_clu_glob_corr_vec True --stride_lay3 2 --stride_lay4 2
    ;;
  "deconv")
    python trainVal.py -c model_cub9.config --model_id deconv  --resnet_bilinear True --bil_cluster True  --bil_cluster_lay_ind 4 --bil_clu_deconv True --stride_lay3 2 --stride_lay4 2
    ;;
  "multiple_stride")
    python trainVal.py -c model_cub9.config --model_id multStr  --resnet_bilinear True --bil_cluster True  --bil_cluster_lay_ind 4 --multiple_stride True --stride_lay3 2 --stride_lay4 2
    ;;
  "clusDil")
    python trainVal.py -c model_cub9.config --model_id clusDil   --resnet_bilinear True --bil_cluster True --resnet_dilation 2 --val_batch_size 120 --max_batch_size 50
    ;;
  "clusEnsRed")
    python trainVal.py -c model_cub9.config --model_id clusEnsRed   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble True  --stride_lay3 2 --stride_lay4 2
    ;;
  "clusRedMultStrInit")
    python trainVal.py -c model_cub9.config --model_id clusRedMultStrInit   --resnet_bilinear True --bil_cluster True --multiple_stride True  --stride_lay3 2 --stride_lay4 2 --init_path ../models/CUB9/modelclusRed_best_epoch43 --epochs 10 --val_batch_size 100 --max_batch_size 50
    ;;
  "clus-initRed")
    python trainVal.py -c model_cub9.config --model_id clus_initRed   --resnet_bilinear True --bil_cluster True --init_path ../models/CUB9/modelclusRed_best_epoch43 --max_batch_size 65 --val_batch_size 100
    ;;
  "clusZoomAct")
    python trainVal.py -c model_cub9.config --model_id clus_zoomAct   --resnet_bilinear True --bil_cluster True --max_batch_size 130 --zoom_on_act True --val_batch_size 100
    ;;
  "clusRedZoomSal")
    python trainVal.py -c model_cub9.config --model_id clusRed_zoomSal   --resnet_bilinear True --bil_cluster True --stride_lay3 2 --stride_lay4 2 --saliency_crop True --max_batch_size 130 --num_workers 12
    ;;
  "clusRedZoomSal-init")
    python trainVal.py -c model_cub9.config --model_id clusRed_zoomSal_init   --resnet_bilinear True --bil_cluster True --stride_lay3 2 --stride_lay4 2 --saliency_crop True --max_batch_size 130 --num_workers 12 --init_path ../models/CUB9/modelclusRed_best_epoch43
    ;;
  "clusRedZoomSalHead-init")
    python trainVal.py -c model_cub9.config --model_id clusRed_zoomSalHead_init   --resnet_bilinear True --bil_cluster True --stride_lay3 2 --stride_lay4 2 --saliency_crop True --max_batch_size 130 --num_workers 12 --init_path ../models/CUB9/modelclusRed_best_epoch43
    ;;
  "clusRedRandZoomSal")
    python trainVal.py -c model_cub9.config --model_id clusRed_randZoomSal   --resnet_bilinear True --bil_cluster True --stride_lay3 2 --stride_lay4 2 --saliency_crop True --random_sal_crop True --max_batch_size 130 --num_workers 12
    ;;
  "clusRed_veryBig")
    python trainVal.py -c model_cub9.config --model_id clusRed_veryBig --very_big_images True  --resnet_bilinear True --bil_cluster True --stride_lay3 2 --stride_lay4 2 --max_batch_size 130
    ;;
  "*")
    echo "no such model"
    ;;
esac
