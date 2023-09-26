case $1 in
  "noneRed2_r50")
    python trainVal.py -c grade_dl4ivf.config --model_id noneRed2_r50 --first_mod resnet50 --big_images True --batch_size 64 --val_batch_size 512 --start_mode scratch $2
    ;;
  "noneRed2_r50_multicenter")
    python trainVal.py -c grade_multicenter.config --model_id noneRed2_r50 --first_mod resnet50 --big_images True --batch_size 64 --val_batch_size 512 --start_mode scratch $2
    ;;
  "*")
    echo "no such model"
    ;;
esac