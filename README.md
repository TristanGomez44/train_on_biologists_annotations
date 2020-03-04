# Process an image as a point-cloud


## Install dependencies

virtualenv ~/pn_env
source ~/pn_env/bin/activate

pip install torch_cluster torch_scatter torch_sparse torch_geometric
pip install torchvision

pip install albumentations

## Train a model

`
python trainVal.py -c model.config --first_mod resnet18 --second_mod pointnet2
`
