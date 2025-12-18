#!/bin/bash
# Script to rerun all RAG experiments with fixed implementations

set -e
cd "$(dirname "$0")"

# Activate virtual environment
source rag_venv/bin/activate

# Set environment variables
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="$PWD/.mplcache"

# Create logs directory
mkdir -p logs

# Set SERPAPI_KEY for CorrectiveRAG (if available)
if [ -z "$SERPAPI_KEY" ]; then
    echo "Warning: SERPAPI_KEY not set. CorrectiveRAG may fall back to googlesearch."
fi

# Dataset path
DATASET="class_files_df.pkl"
OUTPUT_DIR="results"

# Methods to run
METHODS=("simple" "code_aware" "corrective" "fusion" "self")

echo "=========================================="
echo "Rerunning All RAG Experiments"
echo "=========================================="
echo ""
echo "Fixes applied:"
echo "  1. Fixed pydocstyle scoring (exponential decay, filters irrelevant errors)"
echo "  2. Fixed CorrectiveRAG timeouts (60s max web search, strict per-request timeouts)"
echo ""
echo "Methods to run: ${METHODS[@]}"
echo ""
echo "Starting experiments..."
echo ""

# Run each method sequentially
for METHOD in "${METHODS[@]}"; do
    echo "=========================================="
    echo "Running $METHOD RAG"
    echo "=========================================="
    
    LOG_FILE="logs/${METHOD}_rag_rerun_$(date +%Y%m%d_%H%M%S).log"
    
    python run_rag_experiment.py \
        --dataset_path "$DATASET" \
        --single_method "$METHOD" \
        --output_dir "$OUTPUT_DIR" \
        2>&1 | tee "$LOG_FILE"
    
    if [ ${PIPESTATUS[0]} -eq 0 ]; then
        echo ""
        echo "✅ $METHOD RAG completed successfully"
        echo "   Log saved to: $LOG_FILE"
    else
        echo ""
        echo "❌ $METHOD RAG failed"
        echo "   Check log: $LOG_FILE"
        exit 1
    fi
    
    echo ""
done

echo "=========================================="
echo "All Experiments Completed!"
echo "=========================================="
echo ""
echo "Results saved to: $OUTPUT_DIR/"
echo "Evaluations saved to: $OUTPUT_DIR/evaluation/"
echo "Visualizations saved to: $OUTPUT_DIR/visualization/"
echo ""
echo "Next steps:"
echo "  1. Run comparison/aggregation scripts"
echo "  2. Review updated pydocstyle scores"
echo "  3. Check CorrectiveRAG performance improvements"

