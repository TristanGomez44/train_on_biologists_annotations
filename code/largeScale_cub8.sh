case $1 in
  "clusCat")
    python trainVal.py -c model_cub8.config --model_id bilClusCat     --resnet_bilinear True --bil_cluster True
    ;;
  "noneRed")
    python trainVal.py -c model_cub8.config --model_id noneRed          --resnet_layer_size_reduce True
    ;;
  "none")
    python trainVal.py -c model_cub8.config --model_id none --epochs 200
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
    python trainVal.py -c model_cub8.config --model_id bilRed             --resnet_bilinear True   --resnet_layer_size_reduce True
    ;;
  "bil")
    python trainVal.py -c model_cub8.config --model_id bil             --resnet_bilinear True
    ;;
  "clusCatRed")
    python trainVal.py -c model_cub8.config --model_id bilClusCatRed     --resnet_bilinear True --bil_cluster True  --bil_cluster_norefine True  --resnet_layer_size_reduce True
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
  "*")
    echo "no such model"
    ;;
esac
