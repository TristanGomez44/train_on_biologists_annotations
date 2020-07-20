case $1 in
  "bil")
    python trainVal.py -c model_emb.config --exp_id EMB --model_id bil        --epochs 300  --aux_model False  --aux_mod_nll_weight 0 --first_mod $2    --resnet_bilinear True  --resnet_simple_att_score_pred_act_func relu
    ;;
  "clusCat")
    python trainVal.py -c model_emb.config --exp_id EMB --model_id bilClusCat --epochs 300  --aux_model False --aux_mod_nll_weight 0 --first_mod $2   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False
    ;;
  "clusNoGate")
    python trainVal.py -c model_emb.config --exp_id EMB --model_id bilClusNoG --epochs 300  --aux_model False --aux_mod_nll_weight 0 --first_mod $2   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble True
    ;;
  "clus")
    python trainVal.py -c model_emb.config --exp_id EMB --model_id bilClus    --epochs 300  --aux_model False --aux_mod_nll_weight 0 --first_mod $2   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble True --bil_cluster_ensemble_gate True
    ;;
  "none")
    python trainVal.py -c model_emb.config --exp_id EMB --model_id none       --epochs 300  --aux_model False --aux_mod_nll_weight 0 --first_mod $2   --resnet_layer_size_reduce True --aux_model False --aux_mod_nll_weight 0
    ;;
  "noneNoRed")
    python trainVal.py -c model_emb.config --exp_id EMB --model_id none_noRed --epochs 1000 --aux_model False --aux_mod_nll_weight 0 --first_mod $2
    ;;
  "*")
    echo "no such model"
    ;;
esac
