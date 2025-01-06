#!/bin/bash 
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --job-name=GANgBran
#SBATCH --cpus-per-task 1
#SBATCH --mail-type=All
#SBATCH --mail-user=s4653241@uqconnect.edu.au
#SBATCH -o GAN_test_out.txt
#SBATCH -e GAN_test_err.txt
#SBATCH --partition=a100-test
#SBATCH --gres=gpu:1

conda activate Torched

python ~/GAN_Brain/train.py

