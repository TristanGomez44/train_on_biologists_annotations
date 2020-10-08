case $1 in
  "aircraft")
    python trainVal.py -c model_cub8.config --exp_id AIR8 --model_id bilClusCat  \
            --resnet_bilinear True --bil_cluster True  \
            --dataset_train aircraft_train --dataset_val aircraft_train --dataset_test aircraft_test --optuna True
    ;;
  "dogs")
    python trainVal.py -c model_cub8.config --exp_id DOGS8 --model_id bilClusCat  \
            --resnet_bilinear True --bil_cluster True  \
            --dataset_train dogs_train --dataset_val dogs_train --dataset_test dogs_test --optuna True
    ;;
  "cars")
    python trainVal.py -c model_cub8.config --exp_id CARS8 --model_id bilClusCat  \
            --resnet_bilinear True --bil_cluster True  \
            --dataset_train cars_train --dataset_val cars_train --dataset_test cars_test --optuna True
    ;;
  "aircraftBil")
    python trainVal.py -c model_cub8.config --exp_id AIR8 --model_id bil  \
            --resnet_bilinear True --bil_cluster False  \
            --dataset_train aircraft_train --dataset_val aircraft_train --dataset_test aircraft_test --optuna True
    ;;
  "dogsBil")
    python trainVal.py -c model_cub8.config --exp_id DOGS8 --model_id bil  \
            --resnet_bilinear True --bil_cluster False \
            --dataset_train dogs_train --dataset_val dogs_train --dataset_test dogs_test --optuna True
    ;;
  "carsBil")
    python trainVal.py -c model_cub8.config --exp_id CARS8 --model_id bil  \
            --resnet_bilinear True --bil_cluster False \
            --dataset_train cars_train --dataset_val cars_train --dataset_test cars_test --optuna True
    ;;
  "*")
    echo "no such model"
    ;;
esac
