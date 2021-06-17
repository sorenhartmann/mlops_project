 #!/bin/sh
 #BSUB -q gpuv100
 #BSUB -gpu "num=1:mode=exclusive_process"
 #BSUB -J ConvBERT-hparams
 #BSUB -n 8
 #BSUB -W 24:00
 #BSUB -B
 #BSUB -N
 #BSUB -R span[hosts=1]
 #BSUB -R "select[gpu32gb]"
 #BSUB -R "rusage[mem=4GB]"
 #BSUB -o logs/%J.out
 #BSUB -e logs/%J.err

module load python3/3.8.9
module load cuda/8.0
module load cudnn/v7.0-prod-cuda8

.venv/bin/python src/models/search_hparams.py --max_epochs=50 --gpus=1