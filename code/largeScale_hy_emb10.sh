case $1 in
  "clus_mast")
    python trainVal.py -c model_emb10.config --model_id clus_mast   --resnet_bilinear True --bil_cluster True --max_batch_size 130 --val_batch_size 20 --master_net True --m_model_id clusRed --max_batch_size_single_pass 9 
    ;;
  "clusRed")
    python trainVal.py -c model_emb10.config --model_id clusRed     --resnet_bilinear True --bil_cluster True --stride_lay3 2 --stride_lay4 2 --max_batch_size 130 --val_batch_size 40 --max_batch_size_single_pass 25
    ;;
  "bilRed")
    python trainVal.py -c model_emb10.config --model_id bilRed     --resnet_bilinear True   --drop_last False                  --stride_lay3 2 --stride_lay4 2 --max_batch_size 130 --val_batch_size 40 --max_batch_size_single_pass 40 
    ;;
  "noneRed")
    python trainVal.py -c model_emb10.config --model_id noneRed                                               --stride_lay3 2 --stride_lay4 2 --max_batch_size 130 --val_batch_size 40 --max_batch_size_single_pass 25 
    ;;
  "inter_by_parts")
    python trainVal.py -c model_emb10.config --model_id interbyparts   --inter_by_parts True --val_batch_size 400 --max_batch_size_single_pass 80 \
                --stride_lay3 2 --stride_lay4 2  --max_batch_size 130 --always_sched True --drop_last False \
                --strict_init False --start_mode fine_tune --init_path ../models/EMB10/modelnoneRed_best_epoch4 --epochs 6 
    ;;
  "abn")
    python trainVal.py -c model_emb10.config --model_id abn   --abn True --val_batch_size 1200 --max_batch_size_single_pass 130 \
                --stride_lay3 2 --stride_lay4 2  --max_batch_size 130 --always_sched True --drop_last False --epochs 6 --big_images False 
    ;;
  "*")
    echo "no such model"
    ;;
esac
