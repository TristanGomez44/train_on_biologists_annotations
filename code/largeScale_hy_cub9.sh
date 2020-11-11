case $1 in
  "clus")
    python trainVal.py -c model_cub9.config --model_id clus   --resnet_bilinear True --bil_cluster True
    ;;
  "bil")
    python trainVal.py -c model_cub9.config --model_id bil    --resnet_bilinear True
    ;;
  "none")
    python trainVal.py -c model_cub9.config --model_id none
    ;;
  "clusRed")
    python trainVal.py -c model_cub9.config --model_id clusRed     --resnet_bilinear True --bil_cluster True --stride_lay3 2 --stride_lay4 2 --max_batch_size 130
    ;;
  "bilRed")
    python trainVal.py -c model_cub9.config --model_id bilRed      --resnet_bilinear True   --stride_lay3 2 --stride_lay4 2 --max_batch_size 130
    ;;
  "noneRed")
    python trainVal.py -c model_cub9.config --model_id noneRed      --stride_lay3 2 --stride_lay4 2 --max_batch_size 130
    ;;
  "*")
    echo "no such model"
    ;;
esac
