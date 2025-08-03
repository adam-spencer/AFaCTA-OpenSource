#!/bin/bash

# --- Configuration ---
SLURM_SCRIPT="hpc/run-afacta.sh"
MODELS=(
    "phi3:14b-medium-128k-instruct-q8_0"
    "deepseek-r1:8b-llama-distill-fp16"
    "deepseek-r1:14b-qwen-distill-q8_0"
    "gemma3:27b-it-qat"
    "deepseek-r1:8b-0528-qwen3-fp16"
    "phi4:14b-q8_0"
    "qwen2.5:7b-instruct-fp16"
    "llama3.1:8b-instruct-fp16"
    "falcon3:10b-instruct-fp16"
    "mistral:7b-instruct-v0.3-fp16"
    "qwen3:8b-fp16"
    "gemma3:12b-it-fp16"
)

# --- Argument Parsing ---
FILE_PATH=""
TWITTER_FLAG=""

# Loop through all command-line arguments to find the flag and file path
for arg in "$@"; do
    case $arg in
        -t)
        TWITTER_FLAG="-t"
        ;;
        *)
        # Assume any other argument is the file path
        FILE_PATH="$arg"
        ;;
    esac
done

# Check if a file path was provided
if [ -z "$FILE_PATH" ]; then
    echo "Error: No file path specified."
    echo "Usage: $0 <path/to/file.csv> [-t]"
    exit 1
fi


# --- Submission Loop ---
echo "Starting batch submission..."

for model in "${MODELS[@]}"; do
    echo "Submitting job for model: '$model' on file: '$FILE_PATH' (Flag: '$TWITTER_FLAG')"
    
    # Pass the flag to the SLURM script.
    # If -t was not provided, $TWITTER_FLAG will be empty.
    sbatch "$SLURM_SCRIPT" "$model" "$FILE_PATH" "$TWITTER_FLAG"
done

echo "All jobs submitted successfully."
