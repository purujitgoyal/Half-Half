#!/bin/bash
#
#SBATCH --job-name=halfhalf
#SBATCH --output=logsglore/hh_%j.txt  # output file
#SBATCH -e logsglore/hh_%j.err        # File to which STDERR will be written
#SBATCH --partition=2080ti-long # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#
#SBATCH --ntasks=1

python -u glore_train.py ./Data/visual_genome/visual_genome_train_ann.csv ./Data/visual_genome/visual_genome_val_ann.csv /mnt/nfs/scratch1/abajaj/halfhalf/ 100

exit
