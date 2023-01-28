case $1 in
  "noneRed")
    python apply_model_on_video.py -c model_emb10.config --model_id noneRed                                           --stride_lay3 2 --stride_lay4 2 --val_batch_size 40 --img_folder $2
    ;;
  "clus_mast")
    python apply_model_on_video.py -c model_emb10.config --model_id clus_mast   --resnet_bilinear True --bil_cluster True --val_batch_size 20 --img_folder $2
    ;;
  "clus_mast_hung")
    python apply_model_on_video.py -c model_emb10.config --model_id clus_mast   --resnet_bilinear True --bil_cluster True --val_batch_size 20 --img_folder $2 --hungarian_alg --cosine_sim_thres $3
    ;;
  "clus_mast_hung_att_sim")
    python apply_model_on_video.py -c model_emb10.config --model_id clus_mast   --resnet_bilinear True --bil_cluster True --val_batch_size 20 --img_folder $2 --hungarian_alg --eucl_dist_thres $3 --att_maps_sim 
    ;;
  "*")
    echo "no such model"
    ;;
esac


