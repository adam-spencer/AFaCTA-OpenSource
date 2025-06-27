#!/bin/bash
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=aspencer2@sheffield.ac.uk
#SBATCH --mail-type=ALL

module load Anaconda3/2022.05
module load cuDNN/8.9.2.26-CUDA-12.1.1

source activate afacta

export OLLAMA_MODELS=/mnt/parscratch/users/acr24as/ollama_models

echo "Starting Ollama server..."

OLLAMA_SIF_PATH="/${HOME}/ollama-container/ollama.sif"
apptainer instance start --nv "$OLLAMA_SIF_PATH" "ollama-job-${SLURM_JOB_ID}"

echo "Waiting for Ollama server to start..."
sleep 15

echo "Running main script..."
python code/afacta_multi_step_annotation.py \
  --file_name data/raw_speeches/AK1995_processed.csv \
  --output_name AK1995 \
  --llm_name gemma3:12b \
  --context 1

echo "Main script finished, cleaning up..."
apptainer instance stop "ollama-job-${SLURM_JOB_ID}"

echo "Job finished successfully"
