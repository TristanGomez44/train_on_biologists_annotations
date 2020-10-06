case $1 in
  "clusCatSoftm")
    python trainVal.py -c model_cub8.config --dataset_train embryo_img_train --dataset_val embryo_img_train --dataset_test embryo_img_test \
                          --exp_id EMB8 --model_id bilClusCat_softm_hyp --epochs 300      --resnet_bilinear True --bil_cluster True  \
                          --val_batch_size 25 --apply_softmax_on_sim True --softm_coeff 1 --shufle_test_set True
    ;;
  "*")
    echo "no such model"
    ;;
esac
