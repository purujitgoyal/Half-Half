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

python -u test_model.py ./Data/val_ann_encoded.csv ../../Data ./model/model_baseline
#sleep 1
exit

