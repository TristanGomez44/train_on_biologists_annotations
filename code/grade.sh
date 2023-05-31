case $1 in
  "noneRed2_lr_swin")
    python trainVal.py -c grade.config --model_id noneRed2_swin --epochs 100  --max_worse_epoch_nb 10 --first_mod swin_b_16  --big_images False --batch_size 64 --val_batch_size 128
    ;;
  "noneRed2_lr_swin_reg_to_class")
    python trainVal.py -c grade.config --model_id noneRed2_swin_reg_to_class --epochs 100  --max_worse_epoch_nb 10 --first_mod swin_b_16  --big_images False --batch_size 64 --val_batch_size 128 --regression_to_classif True
    ;;
  "noneRed2_lr_swin2")
    python trainVal.py -c grade.config --model_id noneRed2_swin2  --first_mod swin_b_16  --big_images False --batch_size 64 --val_batch_size 128
    ;;
 "noneRed2_lr_swin_regression")
    python trainVal.py -c grade.config --model_id noneRed2_swin_regression  --first_mod swin_b_16  --big_images False --batch_size 64 --val_batch_size 128 --regression True
    ;;
  "noneRed2_lr_r50")
    python trainVal.py -c grade.config --model_id noneRed2_lr_r50 --first_mod resnet50 --big_images False --batch_size 64 --val_batch_size 512
    ;;
  "noneRed2_r50")
    python trainVal.py -c grade.config --model_id noneRed2_r50 --first_mod resnet50 --big_images True --batch_size 64 --val_batch_size 512
    ;;
  "noneRed2_lr_swin_ssl")
    python trainVal.py -c ssl.config --model_id noneRed2_swin --epochs 100  --max_worse_epoch_nb 10 --first_mod swin_b_16  --big_images False --batch_size 64 --val_batch_size 128 --ssl True
    ;;
  "noneRed2_lr_swin_authordataaug_sched")
    python trainVal.py -c grade.config --model_id noneRed2_swin_authordataaug_sched --first_mod swin_b_16  --big_images False --batch_size 64 --val_batch_size 128 --swa True
    ;;
  "noneRed2_lr_swin_one_feat_per_head_relu")
    python trainVal.py -c grade.config --model_id noneRed2_swin_one_feat_per_head_relu --first_mod swin_b_16  --big_images False --batch_size 64 --val_batch_size 128 --one_feat_per_head True
    ;;
  "debug")
    python trainVal.py -c grade.config --model_id noneRed2_lr_r18 --epochs 15 --first_mod resnet18  --big_images False --start_mode scratch --one_feat_per_head True --regression_to_classif True
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