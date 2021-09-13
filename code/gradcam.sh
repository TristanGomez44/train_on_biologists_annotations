
python trainVal.py -c model_cub9.config --model_id clus_masterClusRed   --resnet_bilinear True --bil_cluster True --grad_cam True --exp_id CUB10 --optuna False
python trainVal.py -c model_cub9.config --model_id noneRed  --grad_cam 4816 4835 753  --exp_id CUB10  --optuna False --stride_lay3 2 --stride_lay4 2 --optuna False
python trainVal.py -c model_air10.config --model_id noneRed  --grad_cam 2023 2670 3284  --exp_id AIR10  --optuna False --stride_lay3 2 --stride_lay4 2 --optuna False
python trainVal.py -c model_cars10.config --model_id noneRed  --grad_cam 1974 3136 7209 --exp_id CARS10  --optuna False --stride_lay3 2 --stride_lay4 2 --optuna False

#For teaser
python trainVal.py -c model_cub9.config --model_id noneRed  --grad_cam 122  --exp_id CUB10  --optuna False --stride_lay3 2 --stride_lay4 2 --optuna False
python trainVal.py -c model_cars10.config --model_id noneRed  --grad_cam 6879 --exp_id CARS10  --optuna False --stride_lay3 2 --stride_lay4 2 --optuna False

#Comp 
python trainVal.py -c model_cub10.config --model_id noneRed  --grad_cam 4816 4835 753  --exp_id CUB10  --optuna False --stride_lay3 2 --stride_lay4 2 --optuna False
python trainVal.py -c model_cub10.config --model_id noneRed  --grad_cam 4816 4835 753  --exp_id CUB10  --optuna False --stride_lay3 2 --stride_lay4 2 --optuna False --rise True
python trainVal.py -c model_cub10.config --model_id noneRed  --grad_cam 4816 4835 753  --exp_id CUB10  --optuna False --stride_lay3 2 --stride_lay4 2 --optuna False --score_map True
python trainVal.py -c model_cub10.config --model_id noneRed  --grad_cam 4816 4835 753  --exp_id CUB10  --optuna False --stride_lay3 2 --stride_lay4 2 --optuna False --noise_tunnel True

python trainVal.py -c model_air10.config --model_id noneRed  --grad_cam 2023 2670 3284  --exp_id AIR10  --optuna False --stride_lay3 2 --stride_lay4 2 --optuna False
python trainVal.py -c model_air10.config --model_id noneRed  --grad_cam 2023 2670 3284  --exp_id AIR10  --optuna False --stride_lay3 2 --stride_lay4 2 --optuna False --rise True
python trainVal.py -c model_air10.config --model_id noneRed  --grad_cam 2023 2670 3284  --exp_id AIR10  --optuna False --stride_lay3 2 --stride_lay4 2 --optuna False --score_map True
python trainVal.py -c model_air10.config --model_id noneRed  --grad_cam 2023 2670 3284  --exp_id AIR10  --optuna False --stride_lay3 2 --stride_lay4 2 --optuna False --noise_tunnel True

python trainVal.py -c model_cars10.config --model_id noneRed  --grad_cam 1974 3136 7209 --exp_id CARS10  --optuna False --stride_lay3 2 --stride_lay4 2 --optuna False
python trainVal.py -c model_cars10.config --model_id noneRed  --grad_cam 1974 3136 7209 --exp_id CARS10  --optuna False --stride_lay3 2 --stride_lay4 2 --optuna False --rise True
python trainVal.py -c model_cars10.config --model_id noneRed  --grad_cam 1974 3136 7209 --exp_id CARS10  --optuna False --stride_lay3 2 --stride_lay4 2 --optuna False --score_map True
python trainVal.py -c model_cars10.config --model_id noneRed  --grad_cam 1974 3136 7209 --exp_id CARS10  --optuna False --stride_lay3 2 --stride_lay4 2 --optuna False --noise_tunnel True