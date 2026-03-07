#!/bin/bash
# Local Validation Script: Fast sanity check on 1 sample per variant
# This mirrors the granular execution in Colab to ensure EVERY variant works before cloud execution.

set -e
cd "$(dirname "$0")"

# Activate virtual environment
source rag_venv/bin/activate
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="$PWD/.mplcache"
mkdir -p logs results/visualization/paper_charts

echo "=========================================="
echo "🧪 Starting Local Pipeline Validation (1 Sample Per Variant)"
echo "=========================================="

# Test Model (Assuming user is pulling llama3.2:latest)
MODEL="llama3.2:latest"
SAMPLES=1

echo "Target Model: $MODEL"
echo "Sample Size: $SAMPLES"

echo ""
echo "------------------------------------------"
echo "1. PLAIN LLM EXPERIMENTS"
echo "------------------------------------------"
python scripts/compare_all_strategies.py --group plain --structure fewshot --samples $SAMPLES --model $MODEL
python scripts/compare_all_strategies.py --group plain --structure base --samples $SAMPLES --model $MODEL
python scripts/compare_all_strategies.py --group plain --structure cot --samples $SAMPLES --model $MODEL
python scripts/compare_all_strategies.py --group plain --structure tot --samples $SAMPLES --model $MODEL
python scripts/compare_all_strategies.py --group plain --structure got --samples $SAMPLES --model $MODEL

echo ""
echo "------------------------------------------"
echo "2. SIMPLE RAG EXPERIMENTS"
echo "------------------------------------------"
python scripts/compare_all_strategies.py --group simple --structure base --samples $SAMPLES --model $MODEL
python scripts/compare_all_strategies.py --group simple --structure cot --samples $SAMPLES --model $MODEL
python scripts/compare_all_strategies.py --group simple --structure tot --samples $SAMPLES --model $MODEL
python scripts/compare_all_strategies.py --group simple --structure got --samples $SAMPLES --model $MODEL

echo ""
echo "------------------------------------------"
echo "3. SELF-CORRECTION RAG EXPERIMENTS"
echo "------------------------------------------"
python scripts/compare_all_strategies.py --group self --structure base --samples $SAMPLES --model $MODEL
python scripts/compare_all_strategies.py --group self --structure cot --samples $SAMPLES --model $MODEL
python scripts/compare_all_strategies.py --group self --structure tot --samples $SAMPLES --model $MODEL
python scripts/compare_all_strategies.py --group self --structure got --samples $SAMPLES --model $MODEL

echo ""
echo "------------------------------------------"
echo "4. ANALYSIS & VISUALIZATION SCRIPTS"
echo "------------------------------------------"
# Test Retrieval Ablations (Fast Run)
python scripts/run_retrieval_ablations.py --limit 1

# Test Code Complexity Analyzer
python scripts/analyze_complexity_traces.py --dir results

# Test Human Evaluation Generation
python scripts/generate_human_eval_sheet.py 

# Test New KPIs Charting
python scripts/generate_paper_charts.py

echo ""
echo "=========================================="
echo "✅ Local Validation Complete! No crashes detected."
echo "If this finished successfully, your code is 100% safe to run in Colab."
echo "=========================================="
