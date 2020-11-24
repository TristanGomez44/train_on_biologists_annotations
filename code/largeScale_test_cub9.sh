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
  "clusDil")
    python trainVal.py -c model_cub9.config --model_id clusDil   --resnet_bilinear True --bil_cluster True --resnet_dilation 2 
    ;;
  "*")
    echo "no such model"
    ;;
esac
