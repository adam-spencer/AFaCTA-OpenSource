#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=aspencer2@sheffield.ac.uk
#SBATCH --mail-type=ALL

module load GCC/12.3.0
module load Anaconda3/2022.05
module load cuDNN/8.9.2.26-CUDA-12.1.1

source activate afacta

cd ~/diss/AFaCTA-OpenSource/

echo "Beginning loop..."
for file in data/test_speeches/*.csv; do
  echo "Running on $file"
  python code/afacta_multi_step_annotation.py \
    --file_name "$file" \
    --llm_name /mnt/parscratch/users/acr24as/hf_models/gemma-3-12b-it/
done

echo "Job finished successfully"
