case $1 in
  "noneRed")
    python apply_model_on_video.py -c model_emb10.config --model_id noneRed                                           --stride_lay3 2 --stride_lay4 2 --val_batch_size 40 --img_folder $2
    ;;
  "clus_mast")
    python apply_model_on_video.py -c model_emb10.config --model_id clus_mast   --resnet_bilinear True --bil_cluster True --val_batch_size 20 --img_folder $2
    ;;
  "*")
    echo "no such model"
    ;;
esac
