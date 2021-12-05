#!/bin/bash

# Request a GPU partition node and access to 1 GPU
#SBATCH -p gpu --gres=gpu:1

# Request 1 CPU core
#SBATCH -n 1

#SBATCH --mem=24G
#SBATCH -t 4:00:00
#SBATCH -o sum.out

module load java/8u111                   
module load matlab/R2017b                             
module load intel/2017.0                                  
module load scikit-learn/0.21.2         
module load tensorflow/1.14.0_gpu_py36 
module load cudnn/7.4
module load python/3.6.6_test
module load cuda/10.0.130 
module load keras/2.1.3_py3   

python3 module.py