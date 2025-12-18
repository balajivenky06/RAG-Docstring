#!/bin/bash

# Run SimpleRAG with DeepSeek-R1 (Reasoning Model)
# Using deepseek-r1:1.5b as the generator model
# Saving results to a specific directory

python run_rag_experiment.py \
    --single_method simple \
    --generator_model deepseek-r1:1.5b \
    --output_dir results/reasoning_baseline \
    --dataset_path class_files_df.pkl

echo "Reasoning baseline experiment completed."
