case $1 in
  "bil")
    python trainVal.py -c model_cub4.config --exp_id CUB5 --model_id bil        --epochs 300  --aux_model False  --aux_mod_nll_weight 0 --first_mod $2    --resnet_bilinear True  --resnet_simple_att_score_pred_act_func relu
    ;;
  "bilRed")
    python trainVal.py -c model_cub4.config --exp_id CUB5 --model_id bilRed        --epochs 300  --aux_model False  --aux_mod_nll_weight 0 --first_mod $2    --resnet_bilinear True  --resnet_simple_att_score_pred_act_func relu --resnet_layer_size_reduce True
    ;;
  "bilSmallAtt")
    python trainVal.py -c model_cub4.config --exp_id CUB5 --model_id bilSmallAtt --epochs 300  --aux_model False  --aux_mod_nll_weight 0 --first_mod $2    --resnet_bilinear True  --resnet_simple_att_score_pred_act_func relu --resnet_att_blocks_nb 0
    ;;
  "clusCat")
    python trainVal.py -c model_cub4.config --exp_id CUB5 --model_id bilClusCat --epochs 300  --aux_model False --aux_mod_nll_weight 0 --first_mod $2   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False
    ;;
  "clusCat8")
    python trainVal.py -c model_cub4.config --exp_id CUB5 --model_id bilClusCat_8 --epochs 300  --aux_model False --aux_mod_nll_weight 0 --first_mod $2   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --resnet_bil_nb_parts 8
    ;;
  "clusCat16")
    python trainVal.py -c model_cub4.config --exp_id CUB5 --model_id bilClusCat_16 --epochs 300  --aux_model False --aux_mod_nll_weight 0 --first_mod $2   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --resnet_bil_nb_parts 16
    ;;
  "clusCat32")
    python trainVal.py -c model_cub4.config --exp_id CUB5 --model_id bilClusCat_32 --epochs 300  --aux_model False --aux_mod_nll_weight 0 --first_mod $2   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --resnet_bil_nb_parts 32
    ;;
  "clusCat64")
    python trainVal.py -c model_cub4.config --exp_id CUB5 --model_id bilClusCat_64 --epochs 300  --aux_model False --aux_mod_nll_weight 0 --first_mod $2   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --resnet_bil_nb_parts 64
    ;;
  "clusCatSoftmSched")
    python trainVal.py -c model_cub4.config --exp_id CUB5 --model_id bilClusCatSoftmSched --epochs 300  --aux_model False --aux_mod_nll_weight 0 --first_mod $2   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --bil_clus_soft_sched True
    ;;
  "clusCat5SoftmSched")
    python trainVal.py -c model_cub4.config --exp_id CUB5 --model_id bilClusCat5SoftmSched --epochs 300  --aux_model False --aux_mod_nll_weight 0 --first_mod $2   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --bil_clus_soft_sched True --softm_coeff 5
    ;;
  "clusCatUnorm")
    python trainVal.py -c model_cub4.config --exp_id CUB5 --model_id bilClusCatUnorm --epochs 300  --aux_model False --aux_mod_nll_weight 0 --first_mod $2   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --bil_clust_unnorm True
    ;;
  "clusCatNoRef")
    python trainVal.py -c model_cub4.config --exp_id CUB5 --model_id bilClusCatNoRef --epochs 300  --aux_model False --aux_mod_nll_weight 0 --first_mod $2   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --bil_cluster_norefine True
    ;;
  "clusCatRandVec")
    python trainVal.py -c model_cub4.config --exp_id CUB5 --model_id bilClusCatRandVec --epochs 300  --aux_model False --aux_mod_nll_weight 0 --first_mod $2   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --bil_cluster_randvec True
    ;;
  "clusCatRandVecNoRef")
    python trainVal.py -c model_cub4.config --exp_id CUB5 --model_id bilClusCatRandVecNoRef --epochs 300  --aux_model False --aux_mod_nll_weight 0 --first_mod $2   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --bil_cluster_randvec True  --bil_cluster_norefine True
    ;;
  "clusCatRed")
    python trainVal.py -c model_cub4.config --exp_id CUB5 --model_id bilClusCatRed --epochs 300  --aux_model False --aux_mod_nll_weight 0 --first_mod $2   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --bil_cluster_norefine True  --resnet_layer_size_reduce True
    ;;
  "clusNoGate")
    python trainVal.py -c model_cub4.config --exp_id CUB5 --model_id bilClusNoG --epochs 300  --aux_model False --aux_mod_nll_weight 0 --first_mod $2   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble True
    ;;
  "clus")
    python trainVal.py -c model_cub4.config --exp_id CUB5 --model_id bilClus    --epochs 300  --aux_model False --aux_mod_nll_weight 0 --first_mod $2   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble True --bil_cluster_ensemble_gate True
    ;;
  "none")
    python trainVal.py -c model_cub4.config --exp_id CUB5 --model_id none       --epochs 300  --aux_model False --aux_mod_nll_weight 0 --first_mod $2   --resnet_layer_size_reduce True
    ;;
  "noneNoRed")
    python trainVal.py -c model_cub4.config --exp_id CUB5 --model_id none_noRed --epochs 1000 --aux_model False --aux_mod_nll_weight 0 --first_mod $2
    ;;
  "*")
    echo "no such model"
    ;;
esac
