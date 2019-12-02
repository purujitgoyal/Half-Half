#!/bin/bash
#
#SBATCH --job-name=gcnhalf
#SBATCH --output=logsgcnhalf/hh_%j.txt  # output file
#SBATCH -e logsgcnhalf/hh_%j.err        # File to which STDERR will be written
#SBATCH --partition=titanx-long # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#
#SBATCH --ntasks=1

python -u train_model.py ./Data/train_ann_encoded.csv ./Data/val_ann_encoded.csv ../../Data 100 ./model/model_gcn_baseline ./Data/baseline_glove_word2vec.pkl ./Data/baseline_left_labels.pkl
#sleep 1
exit

