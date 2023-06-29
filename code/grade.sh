case $1 in
  "noneRed2_lr_swin")
    python trainVal.py -c grade.config --model_id noneRed2_swin --epochs 100  --max_worse_epoch_nb 10 --first_mod swin_b_16  --big_images False --batch_size 64 --val_batch_size 50 
    ;;
  "noneRed2_lr_swin_reg_to_class")
    python trainVal.py -c grade.config --model_id noneRed2_swin_reg_to_class --epochs 100  --max_worse_epoch_nb 10 --first_mod swin_b_16  --big_images False --batch_size 64 --val_batch_size 128 --regression_to_classif True
    ;;
  "noneRed2_lr_swin_reg_to_class2")
    python trainVal.py -c grade.config --model_id noneRed2_swin_reg_to_class2 --epochs 100 --first_mod swin_b_16  --big_images False --batch_size 64 --val_batch_size 128 --regression_to_classif True --log_gradient_norm_frequ 1
    ;;
  "noneRed2_lr_swin_reg_to_class3")
    python trainVal.py -c grade.config --model_id noneRed2_swin_reg_to_class3 --epochs 100 --first_mod swin_b_16  --big_images False --batch_size 64 --val_batch_size 128 --regression_to_classif True --log_gradient_norm_frequ 1 --save_output_during_validation True
    ;;
  "noneRed2_lr_swin_reg_to_class3_initrange05")
    python trainVal.py -c grade.config --model_id noneRed2_swin_reg_to_class3_initrange05 --epochs 100 --first_mod swin_b_16  --big_images False --batch_size 64 --val_batch_size 128 --regression_to_classif True --log_gradient_norm_frequ 1 --save_output_during_validation True --init_range_for_reg_to_class_centroid 0.5
    ;;
  "noneRed2_lr_swin2_worse10")
    python trainVal.py -c grade.config --model_id noneRed2_swin2_worse10 --epochs 100  --max_worse_epoch_nb 10 --first_mod swin_b_16  --big_images False --batch_size 64 --val_batch_size 128 --log_gradient_norm_frequ 1
    ;;
 "noneRed2_lr_swin_regression2")
    python trainVal.py -c grade.config --model_id noneRed2_swin_regression2  --first_mod swin_b_16  --big_images False --batch_size 64 --val_batch_size 128 --regression True  --log_gradient_norm_frequ 1
    ;;
 "noneRed2_lr_swin_regression2_nosig")
    python trainVal.py -c grade.config --model_id noneRed2_swin_regression2_nosig  --first_mod swin_b_16  --big_images False --batch_size 64 --val_batch_size 128 --regression True  --log_gradient_norm_frequ 1
    ;;
 "noneRed2_lr_swin_regression")
    python trainVal.py -c grade.config --model_id noneRed2_swin_regression  --first_mod swin_b_16  --big_images False --batch_size 64 --val_batch_size 128 --regression True
    ;;
  "noneRed2_lr_r50")
    python trainVal.py -c grade.config --model_id noneRed2_lr_r50 --first_mod resnet50 --big_images False --batch_size 64 --val_batch_size 512
    ;;
  "none_mast_r50_onefeatperhead")
    python trainVal.py -c grade.config --model_id none_mast_r50_onefeatperhead --first_mod resnet50 --big_images True --batch_size 14 --val_batch_size 32 --one_feat_per_head True --master_net True --m_model_id noneRed2_r50_onefeatperhead_mapsim --stride_lay3 1 --stride_lay4 1 
    ;;
  "none_mast_r50_onefeatperhead_mapsim")
    python trainVal.py -c grade.config --model_id none_mast_r50_onefeatperhead_mapsim --first_mod resnet50 --big_images True --batch_size 14 --val_batch_size 32 --one_feat_per_head True --map_sim_term_weight 1 --master_net True --m_model_id noneRed2_r50_onefeatperhead_mapsim --stride_lay3 1 --stride_lay4 1 
    ;;
  "noneRed2_r50_onefeatperhead_mapsim")
    python trainVal.py -c grade.config --model_id noneRed2_r50_onefeatperhead_mapsim --first_mod resnet50 --big_images True --batch_size 64 --val_batch_size 512 --one_feat_per_head True --map_sim_term_weight 1
    ;;
  "noneRed2_r50_onefeatperhead")
    python trainVal.py -c grade.config --model_id noneRed2_r50_onefeatperhead --first_mod resnet50 --big_images True --batch_size 64 --val_batch_size 512 --one_feat_per_head True
    ;;
  "noneRed2_r50")
    python trainVal.py -c grade.config --model_id noneRed2_r50 --first_mod resnet50 --big_images True --batch_size 64 --val_batch_size 512 --task_to_train icm
    ;;
  "none_mast_r50")
    python trainVal.py -c grade.config --model_id none_mast_r50 --master_net True --m_model_id noneRed2_r50 --first_mod resnet50 --big_images True --batch_size 28 --val_batch_size 128 --task_to_train icm --stride_lay3 1 --stride_lay4 1
    ;;
  "noneRed2_ssl")
    python trainVal.py -c ssl.config --model_id noneRed2_ssl --epochs 1000 --first_mod resnet50  --big_images True --batch_size 16 --val_batch_size 32 --ssl True --dataset_path /LAB-DATA/LS2N/E144069X/DL4IVF/ --num_workers 2
    ;;
  "noneRed2_ssl_noaug")
    python trainVal.py -c ssl.config --model_id noneRed2_ssl_noaug --epochs 1000 --first_mod resnet50  --big_images True --batch_size 16 --val_batch_size 32 --ssl True --dataset_path /LAB-DATA/LS2N/E144069X/DL4IVF/ --num_workers 2 --ssl_data_augment False
    ;;
  "noneRed2_lr_swin_ssl")
    python trainVal.py -c ssl.config --model_id noneRed2_swin_ssl --epochs 100 --first_mod swin_b_16  --big_images False --batch_size 64 --val_batch_size 128 --ssl True
    ;;
  "noneRed2_lr_swin_ssl_ccpl")
    python trainVal.py -c ssl.config --model_id noneRed2_swin_ssl --epochs 100 --first_mod swin_b_16  --big_images False --batch_size 32 --val_batch_size 64 --ssl True --dataset_path /LAB-DATA/LS2N/E144069X/DL4IVF/ --num_workers 2 --val_freq 1
    ;;
  "noneRed2_lr_swin_authordataaug_sched")
    python trainVal.py -c grade.config --model_id noneRed2_swin_authordataaug_sched --first_mod swin_b_16  --big_images False --batch_size 64 --val_batch_size 128 --swa True
    ;;
  "noneRed2_lr_swin_one_feat_per_head_relu")
    python trainVal.py -c grade.config --model_id noneRed2_swin_one_feat_per_head_relu --first_mod swin_b_16  --big_images False --batch_size 64 --val_batch_size 128 --one_feat_per_head True
    ;;
  "noneRed2_lr_swin_one_feat_per_head_relu2")
    python trainVal.py -c grade.config --model_id noneRed2_swin_one_feat_per_head_relu2 --first_mod swin_b_16  --big_images False --batch_size 64 --val_batch_size 128 --one_feat_per_head True --log_gradient_norm_frequ 1
    ;;
  "debug")
    python trainVal.py -c grade.config --model_id noneRed2_lr_r18 --epochs 15 --first_mod resnet18  --big_images False --start_mode scratch --one_feat_per_head True --regression_to_classif True --log_gradient_norm_frequ 1
    ;;
  "debug_ssl")
    /home/E144069X/miniconda3/envs/myenv2/bin/python3 trainVal.py -c ssl.config --model_id noneRed2_lr_r18_ssl --epochs 15 --first_mod resnet18  --big_images False --start_mode scratch --one_feat_per_head True --ssl True --teach_temp_sched_epochs 3 --epochs 6 --dataset_path /media/E144069X/DL4IVF/DL4IVF/ --warmup_epochs 3 --end_cosine_sched_epoch 6
    ;;
  "noneRed2_lr_transf16")
    python trainVal.py -c grade.config --model_id noneRed2_transf16 --epochs 100  --max_worse_epoch_nb 10 --first_mod dit_b_16 --big_images False
    ;;
  "noneRed2_lr_transf8")
    python trainVal.py -c grade.config --model_id noneRed2_transf8 --epochs 100  --max_worse_epoch_nb 10 --first_mod dit_b_8 --big_images False --batch_size 8 --val_batch_size 75
    ;;
  "noneRed2_lr")
    python trainVal.py -c grade.config --model_id noneRed2_lr --epochs 100  --max_worse_epoch_nb 10 --first_mod resnet18 --big_images False --epochs 1
    ;;
  "*")
    echo "no such model"
    ;;
esac