#!/bin/bash

# --- Configuration ---
SLURM_SCRIPT="hpc/run-afacta.sh"
MODELS=(
    "deepseek-r1:14b-qwen-distill-q8_0"
    "deepseek-r1:8b-0528-qwen3-fp16"
    "deepseek-r1:8b-llama-distill-fp16"
    "dolphin3:8b-llama3.1-fp16"
    "falcon3:10b-instruct-fp16"
    "gemma3:12b-it-bf16"
    "gemma3:12b-it-fp16"
    "gemma3:27b-it-qat"
    "llama3.1:70b-instruct-q2_K"
    "llama3.1:8b-instruct-fp16"
    "mistral:7b-instruct-v0.3-fp16"
    "phi3:14b-medium-128k-instruct-q8_0"
    "phi4:14b-q8_0"
    "qwen2.5:14b-instruct-q8_0"
    "qwen2.5:7b-instruct-fp16"
    "qwen3:14b-q8_0"
    "qwen3:8b-fp16"
)

# --- Argument Parsing ---
FILE_PATH=""
declare -a EXTRA_ARGS # Array to hold all optional arguments for sbatch

# This loop processes all arguments passed to this script
while [[ $# -gt 0 ]]; do
  case "$1" in
    -t)
      EXTRA_ARGS+=("-t")
      shift # Move past the flag
      ;;
    --custom_prefix)
      EXTRA_ARGS+=("--custom_prefix" "$2") # Add the flag and its value
      shift # Move past the flag
      shift # Move past the value
      ;;
    --custom_prompts)
      EXTRA_ARGS+=("--custom_prompts" "$2") # Add the flag and its value
      shift # Move past the flag
      shift # Move past the value
      ;;
    *)
      # Assume the first argument not matching a flag is the file path
      if [ -z "$FILE_PATH" ]; then
          FILE_PATH="$1"
      else
          echo "Error: Unknown argument or multiple file paths specified: $1"
          exit 1
      fi
      shift # Move past the argument
      ;;
  esac
done

# Check if a file path was provided
if [ -z "$FILE_PATH" ]; then
    echo "Error: No file path specified."
    echo "Usage: $0 <path/to/file.csv> [-t] [--custom_prefix <prefix>] [--custom_prompts <prompts>]"
    exit 1
fi


# --- Submission Loop ---
echo "Starting batch submission for file: '$FILE_PATH'"
if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    echo "With extra arguments: ${EXTRA_ARGS[@]}"
fi
echo "---"

for model in "${MODELS[@]}"; do
    echo "Submitting job for model: '$model'"
    sbatch "$SLURM_SCRIPT" "$model" "$FILE_PATH" "${EXTRA_ARGS[@]}"
done

echo "---"
echo "All jobs submitted successfully."
