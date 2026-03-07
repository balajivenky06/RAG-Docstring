#!/bin/bash
# Script to run all the tests requested by the reviewers

set -e
cd "$(dirname "$0")"

# Activate virtual environment
source rag_venv/bin/activate
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="$PWD/.mplcache"
mkdir -p logs results/visualization/paper_charts

DATASET="class_files_df.pkl"
echo "=========================================="
echo "Starting Peer Review Revisions Experiments"
echo "=========================================="

# 1. New Few-Shot Baseline Test
echo ""
echo "--- 1. Testing FewShotPlainLLM vs PlainLLM ---"
python scripts/compare_all_strategies.py --group plain --structure fewshot --model llama3.2:latest --samples 100 2>&1 | tee "logs/fewshot_baseline.log"

# 2. Model Generalization (Qwen or DeepSeek)
# User can comment/uncomment these depending on what they have pulled in Ollama
echo ""
echo "--- 2. Checking Model Generalization ---"
# Check if qwen2.5:8b is available.
if ollama list | grep -q 'qwen2.5:8b'; then
    echo "Running Qwen2.5 8B tests..."
    python scripts/compare_all_strategies.py --group simple --structure base --model qwen2.5:8b --samples 100 2>&1 | tee "logs/qwen_generalization.log"
else
    echo "Skipping Qwen2.5 8B setup (model not found). Please pull it manually if you want to test it."
fi

# 3. Retrieval Ablations
echo ""
echo "--- 3. Running Retrieval Ablations (k vs chunk size) ---"
python scripts/run_retrieval_ablations.py --model llama3.2:latest --limit 25 2>&1 | tee "logs/retrieval_ablations.log"

# 4. Generate Human Evaluation Sheet
echo ""
echo "--- 4. Generating Human Evaluation Subset ---"
python scripts/generate_human_eval_sheet.py --dir results/llama3_2_latest 2>&1 | tee "logs/human_eval.log"

# 5. Complexity Breakdown Data
echo ""
echo "--- 5. Analyzing Code Complexity ---"
python scripts/analyze_complexity_traces.py --dir results 2>&1 | tee "logs/complexity_analysis.log"

# 6. Generate the New Beautiful KPI Charts
echo ""
echo "--- 6. Generating New KPI Charts ---"
python scripts/generate_paper_charts.py --dir results/llama3_2_latest 2>&1 | tee "logs/charting.log"

echo ""
echo "--- 7. Running Statistical Significance Tests ---"
python scripts/statistical_analysis.py --dir results/llama3_2_latest 2>&1 | tee "logs/statistical_analysis.log"

echo ""
echo "=========================================="
echo "Revision Tests Complete! Check results/visualization/paper_charts for new plots!"
echo "Statistical significance details printed to logs/statistical_analysis.log"
echo "NOTE: If you haven't filled out the human_eval_sheet.xlsx yet, the correlation plot will be skipped."
echo "=========================================="
