# Process an image as a point-cloud

## Get the data

Create a "data" folder at the root of the project
Download the dataset and put the training and test folders in it.

## Install dependencies

virtualenv ~/pn_env
source ~/pn_env/bin/activate

pip install torch_cluster torch_scatter torch_sparse torch_geometric
pip install torchvision
pip install tensorboardX

## Train a model

`
python trainVal.py -c model.config --first_mod resnet18 --second_mod pointnet2 --exp_id <experience_id> --model_id <model_id>
`

## Check a model performance

`
tensorboard --logdir ../results/<experience_id>
`
