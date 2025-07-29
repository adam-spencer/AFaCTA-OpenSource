#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=4

module load Anaconda3/2022.05

source activate afacta

cd ~/diss/AFaCTA-OpenSource/

echo "Beginning loop..."
for file in data/test_speeches/*.csv; do
  echo "Running on $file"
  python code/afacta_multi_step_sync.py \
    --file_name "$file" \
    --llm_name /mnt/parscratch/users/acr24as/hf_models/gemma-3-12b-it/
done

echo "Job finished successfully"
