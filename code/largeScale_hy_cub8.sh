case $1 in
  "r18")
    python trainVal.py -c model_cub8.config --exp_id CUB8_HYP --model_id r18 --epochs 300   --first_mod resnet18   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False
    ;;
  "r34")
    python trainVal.py -c model_cub8.config --exp_id CUB8_HYP --model_id r34 --epochs 300   --first_mod resnet34   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False
    ;;
  "r101")
    python trainVal.py -c model_cub8.config --exp_id CUB8_HYP --model_id r101 --epochs 300   --first_mod resnet101 --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False
    ;;
  "b3")
    python trainVal.py -c model_cub8.config --exp_id CUB8_HYP --model_id b3 --epochs 300   --first_mod resnet50   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --batch_size 3
    ;;
  "b6")
    python trainVal.py -c model_cub8.config --exp_id CUB8_HYP --model_id b6 --epochs 300   --first_mod resnet50   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --batch_size 6
    ;;
  "b24")
    python trainVal.py -c model_cub8.config --exp_id CUB8_HYP --model_id b24 --epochs 300   --first_mod resnet50   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --batch_size 24
    ;;
  "b48")
    python trainVal.py -c model_cub8.config --exp_id CUB8_HYP --model_id b48 --epochs 300   --first_mod resnet50   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --batch_size 48
    ;;
  "lr0.004")
    python trainVal.py -c model_cub8.config --exp_id CUB8_HYP --model_id lr0.004 --epochs 300   --first_mod resnet50   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --lr 0.004
    ;;
  "lr0.002")
    python trainVal.py -c model_cub8.config --exp_id CUB8_HYP --model_id lr0.002 --epochs 300   --first_mod resnet50   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --lr 0.002
    ;;
  "lr0.0005")
    python trainVal.py -c model_cub8.config --exp_id CUB8_HYP --model_id lr0.0005 --epochs 300   --first_mod resnet50   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --lr 0.0005
    ;;
  "lr0.00025")
    python trainVal.py -c model_cub8.config --exp_id CUB8_HYP --model_id lr0.00025 --epochs 300   --first_mod resnet50   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --lr 0.00025
    ;;
  "lr0.000125")
    python trainVal.py -c model_cub8.config --exp_id CUB8_HYP --model_id lr0.000125 --epochs 300   --first_mod resnet50   --resnet_bilinear True --bil_cluster True --bil_cluster_ensemble False --lr 0.000125
    ;;
  "*")
    echo "no such model"
    ;;
esac
