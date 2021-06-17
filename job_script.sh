#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J ConvBERT
### -- ask for number of cores (default: 1) --
#BSUB -n 1
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=2:mode=exclusive_process"
### Requesting GPU with 32gb memory
#BSUB -R "select[gpu32gb]"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 23:59
# request system-memory
#BSUB -R "rusage[mem=16GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
#BSUB -u s170589@student.dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -oo ConvBERT-%J.out
#BSUB -eo ConvBERT_%J.err
# -- end of LSF options --

nvidia-smi
# Load the cuda module
cd
source VAE-protein/vae_env/bin/activate
cd VAE-protein/Variational_proteins/

python3 gridsearch_ray_tune.py