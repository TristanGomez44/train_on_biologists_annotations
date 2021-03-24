
python trainVal.py -c model_cub9.config --model_id noneRed  --grad_cam 4369 2174 5690 4356 5027 194  --exp_id CUB10  --optuna False --stride_lay3 2 --stride_lay4 2 --optuna False
python trainVal.py -c model_air10.config --model_id noneRed  --grad_cam 744 764 2684 2773 2555 2670  --exp_id AIR10  --optuna False --stride_lay3 2 --stride_lay4 2 --optuna False
python trainVal.py -c model_cars10.config --model_id noneRed  --grad_cam 6296 1332 3843 7910 8123 3427 --exp_id CARS10  --optuna False --stride_lay3 2 --stride_lay4 2 --optuna False
