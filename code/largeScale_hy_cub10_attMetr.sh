case $1 in
  "bilRed")
    python trainVal.py -c model_cub10.config --model_id bilRed_1map      --resnet_bilinear True   --stride_lay3 2 --stride_lay4 2 --max_batch_size 130 \
      --strict_init False --start_mode fine_tune --init_path ../models/CUB10/modelnoneRed_best_epoch34 --max_batch_size_single_pass 20 --resnet_bil_nb_parts 1 --val_batch_size 80 \
      --always_sched True
    ;;
  "clus_mast")
    python trainVal.py -c model_cub10.config --model_id clus_mast_1map   --resnet_bilinear True --bil_cluster True --max_batch_size 130 --val_batch_size 30 --master_net True \
     --m_model_id clusRed --strict_init False --start_mode fine_tune --init_path ../models/CUB10/modelclus_masterClusRed_best_epoch40 --max_batch_size_single_pass 9 \
     --resnet_bil_nb_parts 1 --always_sched True 
    ;;
  "clus_mast_3")
    python trainVal.py -c model_cub10.config --model_id clus_mast_1map_3   --resnet_bilinear True --bil_cluster True --max_batch_size 130 --val_batch_size 30 --master_net True \
                --m_model_id clusRed_3 --max_batch_size_single_pass 9 --resnet_bil_nb_parts 1 --always_sched True --optuna_trial_nb 25
    ;;
  "clusRed")
    python trainVal.py -c model_cub10.config --model_id clusRed_3   --resnet_bilinear True --bil_cluster True --val_batch_size 150 --max_batch_size_single_pass 22 \
                --stride_lay3 2 --stride_lay4 2  --max_batch_size 130 --strict_init False --start_mode fine_tune --init_path ../models/CUB10/modelclusRed_best_epoch43 \
                --resnet_bil_nb_parts 3 --always_sched True --epochs 69
    ;;
  "protoPN")
    python trainVal.py -c model_cub10.config --model_id protopn   --protonet True --val_batch_size 160 --max_batch_size_single_pass 130 \
                --stride_lay3 2 --stride_lay4 2  --max_batch_size 130 --resnet_bil_nb_parts 3 --always_sched True --big_images False --drop_last True \
                --protonet_warm 5 \
                --strict_init False --start_mode fine_tune --init_path ../models/CUB10/modelnoneRed_best_epoch34 --epochs 85
    ;;
  "inter_by_parts")
    #Trained on 2 GPUs without distributed.
    python trainVal.py -c model_cub10.config --model_id interbyparts   --inter_by_parts True --val_batch_size 30 --max_batch_size_single_pass 22 \
                --stride_lay3 2 --stride_lay4 2  --max_batch_size 130 --always_sched True --drop_last False \
                --strict_init False --start_mode fine_tune --init_path ../models/CUB10/modelnoneRed_best_epoch34 --epochs 85
    ;;
  "inter_by_parts_noinit")
    #Trained on 2 GPUs without distributed.
    python trainVal.py -c model_cub10.config --model_id interbyparts_noinit   --inter_by_parts True --val_batch_size 30 --max_batch_size_single_pass 22 \
                --stride_lay3 2 --stride_lay4 2  --max_batch_size 130 --always_sched True 
    ;;
  "prototree")
    #Trained on 2 GPUs without distributed.
    python trainVal.py -c model_cub10.config --model_id prototree   --prototree True --val_batch_size 75 --max_batch_size_single_pass 35 \
                --stride_lay3 2 --stride_lay4 2  --max_batch_size 130 --always_sched True --drop_last True \
                --strict_init False --start_mode fine_tune --init_path ../models/CUB10/modelnoneRed_best_epoch34 --epochs 85 
    ;;
  "*")
    echo "no such model"
    ;;
esac
