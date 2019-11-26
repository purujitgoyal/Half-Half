#!/bin/bash
#
#SBATCH --job-name=halfhalf
#SBATCH --output=logs/hh_%j.txt  # output file
#SBATCH -e logs/hh_%j.err        # File to which STDERR will be written
#SBATCH --partition=titanx-long # Partition to submit to 
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#
#SBATCH --ntasks=1

python -u train_model.py ./Data/train_ann_encoded.csv ./Data/val_ann_encoded.csv ../../Data 100
#sleep 1
exit

