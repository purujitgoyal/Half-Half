#!/bin/bash
#
#SBATCH --job-name=gcnvg
#SBATCH --output=logsgcn/hh_%j.txt  # output file
#SBATCH -e logsgcn/hh_%j.err        # File to which STDERR will be written
#SBATCH --partition=titanx-long # Partition to submit to
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --mem=80G
#
#SBATCH --ntasks=1

python -u gcn_train.py ./Data/visual_genome/visual_genome_train_ann_final.csv ./Data/visual_genome/visual_genome_val_ann_final.csv /mnt/nfs/scratch1/abajaj/halfhalf/ 100 ./model/model_gcn_vg ./Data/visualgenome_glove_word2vec.pkl ./Data/visualgenome_left_labels.pkl

exit
