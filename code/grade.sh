case $1 in
  "noneRed2_r50")
    python trainVal.py -c grade_dl4ivf.config --model_id noneRed2_r50 --first_mod resnet50 --big_images True --batch_size 64 --val_batch_size 512 --start_mode scratch $2
    ;;
  "noneRed2_r50_all")
    python trainVal.py -c grade_dl4ivf.config --model_id noneRed2_r50_all --first_mod resnet50 --big_images True --batch_size 32 --val_batch_size 256 --start_mode scratch --task_to_train all $2
    ;;
  "none_mast2_r50_all")
    python trainVal.py -c grade_dl4ivf.config --model_id none_mast2_r50_all --first_mod resnet50 --big_images True --batch_size 8 --val_batch_size 64 --start_mode auto --task_to_train all --master_net True --m_model_id noneRed2_r50_all --resnet_bilinear True --bil_cluster False --stride_lay3 1 --stride_lay4 1 $2
    ;;
  "bil_mast2_r50_cropped_all")
    python trainVal.py -c grade_dl4ivf.config --model_id bil_mast2_r50_cropped_all --first_mod resnet50 --big_images True --batch_size 8 --val_batch_size 64 --start_mode scratch --task_to_train all --master_net True --m_model_id noneRed2_r50_all --resnet_bilinear True --bil_cluster False --stride_lay3 1 --stride_lay4 1 --train_on_master_masks True $2
    ;;
  "bilRed2_r50_all")
    python trainVal.py -c grade_dl4ivf.config --model_id bilRed2_r50_all --first_mod resnet50 --big_images True --batch_size 32 --val_batch_size 256 --start_mode scratch --task_to_train all --resnet_bilinear True --bil_cluster False $2
    ;;
  "bilRed2_r50_all_moreannot")
    python trainVal.py -c grade_dl4ivf.config --model_id bilRed2_r50_all_moreannot --first_mod resnet50 --big_images True --batch_size 32 --val_batch_size 215 --start_mode scratch --task_to_train all --resnet_bilinear True --bil_cluster False $2
    ;;
  "bilRed2_r50_all_moreannot_null")
    python trainVal.py -c grade_dl4ivf.config --model_id bilRed2_r50_all_moreannot_null --first_mod resnet50 --big_images True --batch_size 32 --val_batch_size 215 --start_mode scratch --task_to_train all --resnet_bilinear True --bil_cluster False $2
    ;;
  "bilRed2_r50_all_moreannot_null_distr")
    python trainVal.py -c grade_dl4ivf.config --model_id bilRed2_r50_all_moreannot_null_distr --first_mod resnet50 --big_images True --batch_size 32 --val_batch_size 215 --start_mode scratch --task_to_train all --resnet_bilinear True --bil_cluster False --distribution_learning True $2
    ;;
  "noneRed2_r50_multicenter")
    python trainVal.py -c grade_multicenter.config --model_id noneRed2_r50 --first_mod resnet50 --big_images True --batch_size 64 --val_batch_size 512 --start_mode scratch $2
    ;;
  "*")
    echo "no such model"
    ;;
esac