case $1 in
  "clusCatSoftm")
    python trainVal.py -c model_cub8.config --dataset_train embryo_img_train --dataset_val embryo_img_train --dataset_test embryo_img_test \
                          --exp_id EMB8 --model_id bilClusCat_softm_hyp --epochs 160      --resnet_bilinear True --bil_cluster True  \
                          --val_batch_size 25 --apply_softmax_on_sim True --softm_coeff 1 --shuffle_test_set True
    ;;
  "clusCatSoftm-aux")
    python trainVal.py -c model_cub8.config --dataset_train embryo_img_train --dataset_val embryo_img_train --dataset_test embryo_img_test \
                          --exp_id EMB8 --model_id bilClusCat_softm_hyp_aux --epochs 50      --resnet_bilinear True --bil_cluster True  \
                          --val_batch_size 25 --apply_softmax_on_sim True --softm_coeff 1 --shuffle_test_set True --aux_on_masked True --lr 0.006 --run_test False
    ;;
  "*")
    echo "no such model"
    ;;
esac
