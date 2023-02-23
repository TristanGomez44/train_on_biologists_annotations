#!/bin/bash
sbatch <<EOT
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

# Nom du fichier de log de Slurm pour le job
#SBATCH -o crohn25_$1.out
#SBATCH -e crohn25_$1.err

source .env/bin/activate
module load python/3.9.4

export CUDA_VISIBLE_DEVICES=0,1

./largeScale_crohn25.sh $1
			   
exit 0
EOT