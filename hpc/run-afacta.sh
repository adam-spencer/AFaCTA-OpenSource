#!/bin/bash
#SBATCH --time=23:59:59
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=aspencer2@sheffield.ac.uk
#SBATCH --mail-type=ALL

# --- Check for required arguments ---
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Error: Missing required arguments."
    echo "Usage: sbatch $0 <model_name> <file_path> [-t]"
    exit 1
fi

# --- Assign arguments to variables ---
MODEL_NAME="$1"
FILE_PATH="$2"
FLAG="$3"  # The optional flag, e.g., -t

# --- Load modules and set environment ---
module load Anaconda3/2022.05
module load cuDNN/8.9.2.26-CUDA-12.1.1
source activate afacta
export OLLAMA_MODELS=/mnt/parscratch/users/acr24as/ollama_models

# --- Start Ollama Server ---
echo "Starting Ollama server..."
OLLAMA_SIF_PATH="/${HOME}/ollama-container/ollama.sif"
INSTANCE_NAME="ollama-job-${SLURM_JOB_ID}"
apptainer instance start --nv "$OLLAMA_SIF_PATH" "$INSTANCE_NAME"
apptainer exec instance://$INSTANCE_NAME bash -c "ollama serve &"
sleep 15

# --- Run Python Script based on flag ---
echo "Running Python script..."
echo "Using model: $MODEL_NAME"
echo "Processing file: $FILE_PATH"

if [ "$FLAG" = "-t" ]; then
    echo "Flag '-t' detected. Running afacta_twitter.py"
    python code/afacta_twitter.py \
        --file_name "$FILE_PATH" \
        --llm_name "$MODEL_NAME"
else
    echo "No '-t' flag. Running afacta_multi_step_annotation.py"
    python code/afacta_multi_step_annotation.py \
        --file_name "$FILE_PATH" \
        --llm_name "$MODEL_NAME"
fi

# --- Stop Ollama Server ---
echo "Main script finished, stopping server..."
apptainer instance stop "$INSTANCE_NAME"

echo "Job finished successfully"
