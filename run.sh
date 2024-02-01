#! /bin/bash

#SBATCH -A MST111109
#SBATCH -J run_main
#SBATCH -p gp4d
#SBATCH -e slurm_txts/run_main.txt
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=gohyixian456@gmail.com

python main.py