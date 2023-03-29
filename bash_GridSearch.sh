#!/bin/bash
#SBATCH --array=1
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 1
#SBATCH --mem=16G
#SBATCH -p batch
#SBATCH -t 4-00:00:00
#SBATCH -o Results/output-%A-%a.out

module load anaconda/2022.05
source /gpfs/runtime/opt/anaconda/2020.02/etc/profile.d/conda.sh
conda activate ddm

cd Codes/test
python3 -u mle_search.py 