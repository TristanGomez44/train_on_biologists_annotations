conda install -y pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=10.2 -c pytorch
conda install -y -c conda-forge matplotlib optuna opencv 
conda install -y  -c anaconda scipy scikit-learn 
conda install -y -c pytorch captum 
pip3 install saliency_maps_metrics --upgrade --no-cache-dir
