#!/bin/sh
#SBATCH --job-name mlp_2_day_memory_y_bid
#SBATCH --error mlp_2_day_memory_y_bid-error.e%j
#SBATCH --output mlp_2_day_memory_y_bid-out.o%j
#SBATCH --partition shared-gpu
#SBATCH --gpus=1
#SBATCH --time 06:00:00
#SBATCH --mem=10000

module load GCCcore/11.2.0 Python/3.9.6
module load GCCcore/10.2.0 CUDA/11.1.1

pip install scikit-learn==1.0.2
pip install numpy==1.21.5
pip install pandas==1.4.4
pip install optuna==3.1.0

python main_tuning.py MLP2DayMemory y_bid