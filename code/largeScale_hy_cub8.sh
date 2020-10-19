case $1 in
  "r18")
    python trainVal.py -c model_cub8.config --exp_id CUB8_HYP --model_id r18 --first_mod resnet18   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False
    ;;
  "r34")
    python trainVal.py -c model_cub8.config --exp_id CUB8_HYP --model_idd resnet34   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False
    ;;
  "r101")
    python trainVal.py -c model_cub8.config --exp_id CUB8_HYP --model_id r101 --first_mod resnet101 --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False
    ;;
  "b3")
    python trainVal.py -c model_cub8.config --exp_id CUB8_HYP --model_id b3 --first_mod resnet50   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --batch_size 3
    ;;
  "b6")
    python trainVal.py -c model_cub8.config --exp_id CUB8_HYP --model_id b6 --first_mod resnet50   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --batch_size 6
    ;;
  "b24")
    python trainVal.py -c model_cub8.config --exp_id CUB8_HYP --model_id b24 --first_mod resnet50   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --batch_size 24
    ;;
  "b24-lr0.004")
    python trainVal.py -c model_cub8.config --exp_id CUB8_HYP --model_id b24-lr0.004 --first_mod resnet50   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --batch_size 24 --lr 0.004
    ;;
  "b12-lr0.002")
    python trainVal.py -c model_cub8.config --exp_id CUB8_HYP --model_id b12-lr0.002 --first_mod resnet50   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --batch_size 12 --lr 0.002
    ;;
  "b48-lr0.008")
    python trainVal.py -c model_cub8.config --exp_id CUB8_HYP --model_id b48-lr0.008 --first_mod resnet50   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --batch_size 48 --lr 0.008
    ;;
  "b48-lr0.008-adam")
    python trainVal.py -c model_cub8.config --exp_id CUB8_HYP --model_id b48-lr0.008-adam --first_mod resnet50   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --batch_size 48 --lr 0.008 --use_scheduler False --optim Adam
    ;;
  "b48-lr0.008-r101")
    python trainVal.py -c model_cub8.config --exp_id CUB8_HYP --model_id b48-lr0.008-r101 --first_mod resnet101   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --batch_size 48 --lr 0.008
    ;;
  "b72-lr0.012")
    python trainVal.py -c model_cub8.config --exp_id CUB8_HYP --model_id b72-lr0.012 --first_mod resnet50   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --batch_size 72 --lr 0.012
    ;;
  "b24-lr0.0005")
    python trainVal.py -c model_cub8.config --exp_id CUB8_HYP --model_id b24-lr0.0005 --first_mod resnet50   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --batch_size 24 --lr 0.0005
    ;;
  "b33")
    python trainVal.py -c model_cub8.config --exp_id CUB8_HYP --model_id b33 --first_mod resnet50   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --batch_size 33
    ;;
  "b16")
    python trainVal.py -c model_cub8.config --exp_id CUB8_HYP --model_id b16 --first_mod resnet50   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --batch_size 16
    ;;
  "b20")
    python trainVal.py -c model_cub8.config --exp_id CUB8_HYP --model_id b20 --first_mod resnet50   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --batch_size 20
    ;;
  "b20-lr0.004")
    python trainVal.py -c model_cub8.config --exp_id CUB8_HYP --model_id b20-lr0.004 --first_mod resnet50   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --batch_size 20 --lr 0.004
    ;;
  "b40-lr0.008")
    python trainVal.py -c model_cub8.config --exp_id CUB8_HYP --model_id b40-lr0.008 --first_mod resnet50   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --batch_size 40 --lr 0.008
    ;;
  "b60-lr0.012")
    python trainVal.py -c model_cub8.config --exp_id CUB8_HYP --model_id b60-lr0.012 --first_mod resnet50   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --batch_size 60 --lr 0.012
    ;;
  "b48")
    python trainVal.py -c model_cub8.config --exp_id CUB8_HYP --model_id b48 --first_mod resnet50   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --batch_size 48
    ;;
  "lr0.004")
    python trainVal.py -c model_cub8.config --exp_id CUB8_HYP --model_id lr0.004 --first_mod resnet50   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --lr 0.004
    ;;
  "lr0.002")
    python trainVal.py -c model_cub8.config --exp_id CUB8_HYP --model_id lr0.002 --first_mod resnet50   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --lr 0.002
    ;;
  "lr0.0005")
    python trainVal.py -c model_cub8.config --exp_id CUB8_HYP --model_id lr0.0005 --first_mod resnet50   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --lr 0.0005
    ;;
  "lr0.00025")
    python trainVal.py -c model_cub8.config --exp_id CUB8_HYP --model_id lr0.00025 --first_mod resnet50   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --lr 0.00025
    ;;
  "lr0.000125")
    python trainVal.py -c model_cub8.config --exp_id CUB8_HYP --model_id lr0.000125 --first_mod resnet50   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --lr 0.000125
    ;;
  "optuna")
    python trainVal.py -c model_cub8.config --exp_id CUB8_HYP --model_id optuna --first_mod resnet50   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --optuna True
    ;;
  "optuna-avg")
    python trainVal.py -c model_cub8.config --exp_id CUB8_HYP --model_id optuna-avg --first_mod resnet50   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --optuna True
    ;;
  "optuna-avg-datAug")
    python trainVal.py -c model_cub8.config --exp_id CUB8_HYP --model_id optuna-avg-dataAug --first_mod resnet50   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --optuna True --opt_data_aug True
    ;;
  "opt-bil")
    python trainVal.py -c model_cub8.config --exp_id CUB8_HYP --model_id opt-bil --first_mod resnet50   --resnet_bilinear True --bil_cluster False --optuna True
    ;;
  "opt-bil-avg")
    python trainVal.py -c model_cub8.config --exp_id CUB8_HYP --model_id opt-bil-avg --first_mod resnet50   --resnet_bilinear True --bil_cluster False --optuna True
    ;;
  "*")
    echo "no such model"
    ;;
esac
