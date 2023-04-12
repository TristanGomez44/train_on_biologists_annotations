#!/bin/bash
#
#
# Partition visee
#SBATCH --partition=GPU-short
#SBATCH --gres=gpu:t4
#
# Nombre de noeuds
#SBATCH --nodes=1
# Nombre de processus MPI par noeud
#SBATCH --ntasks-per-node=4
# Usage reserve des noeuds
#
# Temps de presence du job
#SBATCH --time=24:00:00
#
# Adresse mel de l’utilisateur
#SBATCH --mail-user=tristan.gomez@univ-nantes.fr
# Envoi des mails
#SBATCH --mail-type=fail,abort,end
#SBATCH --exclusive
# Nom du fichier de log de Slurm pour le job
#SBATCH -o model_cub25.config.out
#SBATCH -e model_cub25.config.err

source .env/bin/activate
module load python/3.9.4

export CUDA_VISIBLE_DEVICES=0,1

./largeScale_25.sh noneRed2_lr model_cub25.config
./largeScale_25.sh noneRed_focal2_lr model_cub25.config

./eval_expl_25.sh model_cub25.config all all noneRed2_lr all default
./eval_expl_25.sh model_cub25.config all all noneRed_focal2_lr all default
