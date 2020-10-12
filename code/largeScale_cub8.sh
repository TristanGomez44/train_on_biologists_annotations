case $1 in
  "clusCat")
    python trainVal.py -c model_cub8.config --model_id bilClusCat     --resnet_bilinear True --bil_cluster True --optuna True
    ;;
  "noneRed")
    python trainVal.py -c model_cub8.config --model_id noneRed          --with_seg True --stride_lay3 2 --stride_lay4 2 --optuna True
    ;;
  "none")
    python trainVal.py -c model_cub8.config --model_id none --epochs 300  --with_seg True --optuna True
    ;;
  "clusCatRandVecNoRef")
    python trainVal.py -c model_cub8.config --model_id bilClusCatRandVecNoRef     --resnet_bilinear True --bil_cluster True  --bil_cluster_randvec True  --bil_cluster_norefine True
    ;;
  "clusCatRandVec")
    python trainVal.py -c model_cub8.config --model_id bilClusCatRandVec     --resnet_bilinear True --bil_cluster True  --bil_cluster_randvec True
    ;;
  "clusCatNoRef")
    python trainVal.py -c model_cub8.config --model_id bilClusCatNoRef     --resnet_bilinear True --bil_cluster True  --bil_cluster_norefine True
    ;;
  "bilRed")
    python trainVal.py -c model_cub8.config --model_id bilRed             --resnet_bilinear True   --stride_lay3 2 --stride_lay4 2 --optuna True
    ;;
  "bil")
    python trainVal.py -c model_cub8.config --model_id bil             --resnet_bilinear True --optuna True
    ;;
  "clusCatRed")
    python trainVal.py -c model_cub8.config --model_id bilClusCatRed     --resnet_bilinear True --bil_cluster True --stride_lay3 2 --stride_lay4 2 --optuna True --bil_cluster_norefine True
    ;;
  "clusCatRed-ref")
    python trainVal.py -c model_cub8.config --model_id bilClusCatRed-ref     --resnet_bilinear True --bil_cluster True --stride_lay3 2 --stride_lay4 2 --optuna True
    ;;
  "clusCat-nosqRes")
    python trainVal.py -c model_cub8.config --model_id bilClusCat-nosqRes     --resnet_bilinear True --bil_cluster True --sq_resizing False
    ;;
  "clusCat-01")
    python trainVal.py -c model_cub8.config --model_id bilClusCat_vec01     --resnet_bilinear True --bil_cluster True  --bil_clus_vect_ind_to_use 0,1
    ;;
  "clusCat-12")
    python trainVal.py -c model_cub8.config --model_id bilClusCat_vec12     --resnet_bilinear True --bil_cluster True  --bil_clus_vect_ind_to_use 1,2
    ;;
  "clusCat-0")
    python trainVal.py -c model_cub8.config --model_id bilClusCat_vec0     --resnet_bilinear True --bil_cluster True  --bil_clus_vect_ind_to_use 0
    ;;
  "clusCat-1")
    python trainVal.py -c model_cub8.config --model_id bilClusCat_vec1     --resnet_bilinear True --bil_cluster True  --bil_clus_vect_ind_to_use 1
    ;;
  "clusCat-2")
    python trainVal.py -c model_cub8.config --model_id bilClusCat_vec2     --resnet_bilinear True --bil_cluster True  --bil_clus_vect_ind_to_use 2
    ;;
  "clusCat-aux")
    python trainVal.py -c model_cub8.config --model_id bilClusCat_aux     --resnet_bilinear True --bil_cluster True  --aux_on_masked True
    ;;
  "clusCat-aux-lr0.006")
    python trainVal.py -c model_cub8.config --model_id bilClusCat_aux-lr0.006     --resnet_bilinear True --bil_cluster True  --aux_on_masked True --lr 0.006
    ;;
  "*")
    echo "no such model"
    ;;
esac
