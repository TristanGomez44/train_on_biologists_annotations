
./cross_validation.sh 5 0 80 10 10 model_smallgpu.config
./cross_validation.sh 5 1 80 10 10 model_smallgpu.config

./agregate_cross_validation.sh
