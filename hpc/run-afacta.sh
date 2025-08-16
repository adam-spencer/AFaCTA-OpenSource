#!/bin/bash
#SBATCH --time=12:59:59
#SBATCH --partition=gpu
#SBATCH --qos=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --mem=82G
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=aspencer2@sheffield.ac.uk
#SBATCH --mail-type=ALL

# --- Initialize variables ---
FLAG_T=false
POSITIONAL_ARGS=()
declare -a PY_ARGS # Optional python args

# --- Argument Parsing Loop ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    -t)
      FLAG_T=true
      shift # Move past the '-t' flag
      ;;
    --custom_prefix)
      PY_ARGS+=("--custom_prefix" "$2") # Add flag and its value to the array
      shift # Move past the flag
      shift # Move past the value
      ;;
    --custom_prompts)
      PY_ARGS+=("--custom_prompts" "$2") # Add flag and its value to the array
      shift # Move past the flag
      shift # Move past the value
      ;;
    *)
      # Any other argument is treated as a required positional argument
      POSITIONAL_ARGS+=("$1")
      shift # Move past the argument
      ;;
  esac
done

# --- Assign positional arguments and check for required ones ---
MODEL_NAME="${POSITIONAL_ARGS[0]}"
FILE_PATH="${POSITIONAL_ARGS[1]}"

if [ -z "$MODEL_NAME" ] || [ -z "$FILE_PATH" ]; then
    echo "Error: Missing required arguments."
    echo "Usage: sbatch $0 <model_name> <file_path> [-t] [--custom_prefix <prefix>] [--custom_prompts <prompts>]"
    exit 1
fi


# --- Load modules and set environment ---
module load Anaconda3/2022.05
module load cuDNN/8.9.2.26-CUDA-12.1.1
source activate afacta
export OLLAMA_MODELS=/mnt/parscratch/users/acr24as/ollama_models

# --- Set Ollama port in valid range ---
BASE_PORT=49152
PORT_RANGE=16383 # (65535 - 49152)
OLLAMA_PORT=$((BASE_PORT + SLURM_JOB_ID % PORT_RANGE))

while netstat -tuln | grep -q ":${OLLAMA_PORT}"
do
  echo "Port ${OLLAMA_PORT} is in use, trying next..."
  OLLAMA_PORT=$((OLLAMA_PORT + 1))
done

export OLLAMA_PORT
export OLLAMA_HOST="0.0.0.0:${OLLAMA_PORT}"

# --- Start Ollama Server ---
echo "Starting Ollama server on $OLLAMA_HOST..."
OLLAMA_SIF_PATH="/${HOME}/ollama-container/ollama.sif"
INSTANCE_NAME="ollama-job-${SLURM_JOB_ID}"
apptainer instance start --nv "$OLLAMA_SIF_PATH" "$INSTANCE_NAME"
apptainer exec --env OLLAMA_HOST="${OLLAMA_HOST}" \
  instance://$INSTANCE_NAME bash -c "ollama serve &"
sleep 15

# --- Run Python Script based on flag ---
echo "Running Python script..."
echo "Using model: $MODEL_NAME"
echo "Processing file: $FILE_PATH"

if $FLAG_T; then
    echo "Flag '-t' detected. Running afacta_twitter.py"
    python code/afacta_twitter.py \
        --file_name "$FILE_PATH" \
        --llm_name "$MODEL_NAME" \
        --num-tokens 4096 \
        "${PY_ARGS[@]}"
else
    echo "No '-t' flag. Running afacta_multi_step_annotation.py"
    python code/afacta_multi_step_annotation.py \
        --file_name "$FILE_PATH" \
        --llm_name "$MODEL_NAME" \
        --num-tokens 4096 \
        "${PY_ARGS[@]}"
fi

# --- Stop Ollama Server ---
echo "Main script finished, stopping server..."
apptainer instance stop "$INSTANCE_NAME"

echo "Job finished successfully"
