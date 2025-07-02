#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=1
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

OLLAMA_PID=$!

echo "Waiting for Ollama server to start with PID= $OLLAMA_PID"
sleep 5

echo "Running main script..."
python code/afacta_multi_step_annotation.py \
  --file_name data/raw_speeches/AK1995_processed.csv \
  --output_name AK1995 \
  --llm_name gemma3:12b \
  --context 1

trap 'echo "Script ending. Killing Ollama server (PID: $OLLAMA_PID)..."; kill $OLLAMA_PID' EXIT
