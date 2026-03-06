#!/bin/bash
#SBATCH ...

# Load necessary modules (if applicable on your cluster)
# The following line loads the CUDA module version 12.4 and ignores any cached version.
module --ignore-cache load cuda/12.4
# You can also load additional modules, for example:
# module load anaconda/XXXX.XX

echo "Loading Conda environment..."
source activate /path/to/environment
echo "Conda environment loaded. Starting training..."

python train.py full.yaml

echo "Training script finished."
