case $1 in
  "noneRed2_lr_transf_debug")
    python trainVal.py -c grade.config --model_id noneRed2_transf --epochs 100  --max_worse_epoch_nb 10 --first_mod vit_b_16 --big_images False --epochs 1 --debug True --start_mode scratch 
    ;;
  "noneRed2_lr_transf")
    python trainVal.py -c grade.config --model_id noneRed2_transf --epochs 100  --max_worse_epoch_nb 10 --first_mod vit_b_16 --big_images False --epochs 1
    ;;
  "noneRed2_lr")
    python trainVal.py -c grade.config --model_id noneRed2_lr --epochs 100  --max_worse_epoch_nb 10 --first_mod resnet18 --big_images False --epochs 1
    ;;
  "*")
    echo "no such model"
    ;;
esac

