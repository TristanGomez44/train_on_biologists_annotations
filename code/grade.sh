case $1 in
  "noneRed2_lr_swin_debug")
    python trainVal.py -c grade.config --model_id noneRed2_swin --epochs 100  --max_worse_epoch_nb 10 --first_mod swin_b_16  --big_images False --epochs 1 --debug True  
    ;;
  "noneRed2_lr_swin")
    echo swing
    python trainVal.py -c grade.config --model_id noneRed2_swin --epochs 100  --max_worse_epoch_nb 10 --first_mod swin_b_16  --big_images False --batch_size 64 --val_batch_size 128
    ;;
  "noneRed2_lr_dino_debug")
    python trainVal.py -c grade.config --model_id noneRed2_dino --epochs 100  --max_worse_epoch_nb 10 --first_mod dit_b_16  --big_images False --epochs 1 --debug True  
    ;;
  "noneRed2_lr_dino8_debug")
    python trainVal.py -c grade.config --model_id noneRed2_dino8 --epochs 100  --max_worse_epoch_nb 10 --first_mod dit_b_8  --big_images False --epochs 1 --debug True  
    ;;
  "noneRed2_lr_transf_debug")
    python trainVal.py -c grade.config --model_id noneRed2_transf --epochs 100  --max_worse_epoch_nb 10 --first_mod vit_b_16  --big_images False --epochs 1 --debug True --start_mode scratch 
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

