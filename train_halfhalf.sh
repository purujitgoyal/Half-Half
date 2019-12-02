#!/bin/bash
#
#SBATCH --job-name=glorehalf
#SBATCH --output=logshalf/hh_%j.txt  # output file
#SBATCH -e logshalf/hh_%j.err        # File to which STDERR will be written
#SBATCH --partition=2080ti-long # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#
#SBATCH --ntasks=1

python -u train_model.py ./Data/train_ann_encoded.csv ./Data/val_ann_encoded.csv ../../Data 100 ./model/model_glore_baseline ./data/baseline_glove_word2vec.pkl ./data/baseline_left_labels.pkl
#sleep 1
exit
