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
INSTANCE_NAME="ollama-job-${SLURM_JOB_ID}"
apptainer instance start --nv "$OLLAMA_SIF_PATH" "$INSTANCE_NAME"

apptainer exec instance://$INSTANCE_NAME \
  bash -c "ollama serve &"

sleep 15

models=("gemma3:12b-it-fp16" "llama3.1:8b-instruct-fp16" "falcon3:10b-instruct-fp16" "mistral:7b-instruct-v0.3-fp16" "qwen2.5:7b-instruct-fp16")

echo "Beginning loop..."
for model in ${models[@]}; do
  echo "Using model $model"
  for file in data/test_speeches/*.csv; do
    echo "Running on $file"
    python code/afacta_multi_step_annotation.py \
      --file_name "$file" \
      --llm_name "$model"
  done
done

echo "Main script finished, killing server..."
apptainer instance stop "$INSTANCE_NAME"

echo "Job finished successfully"
