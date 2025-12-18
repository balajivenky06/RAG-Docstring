"""
Script to compare ALL RAG strategies including reasoning variants.
Strategies:
1. SimpleRAG (Base, CoT, ToT, GoT)
2. CodeAwareRAG (Base, CoT, ToT, GoT)
3. CorrectiveRAG (Base, CoT, ToT, GoT)
4. FusionRAG (Base, CoT, ToT, GoT)
5. SelfRAG (Base, CoT, ToT, GoT)
"""

import os
import pandas as pd
import time
import logging
import argparse
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag_system import (
    SimpleRAG, CoTRAG, ToTRAG, GoTRAG,
    SelfCorrectionRAG, CoTSelfCorrectionRAG, ToTSelfCorrectionRAG, GoTSelfCorrectionRAG,
    PlainLLM, CoTPlainLLM, ToTPlainLLM, GoTPlainLLM,
    RAGEvaluator, config
)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Map names to classes
STRATEGY_MAP = {
    # Plain LLM
    "PlainLLM": PlainLLM,
    "CoTPlainLLM": CoTPlainLLM,
    "ToTPlainLLM": ToTPlainLLM,
    "GoTPlainLLM": GoTPlainLLM,

    # Simple
    "SimpleRAG": SimpleRAG,
    "CoTRAG": CoTRAG,
    "ToTRAG": ToTRAG,
    "GoTRAG": GoTRAG,
    

    
    # Self-Correction (formerly SelfRAG)
    "SelfCorrectionRAG": SelfCorrectionRAG,
    "CoTSelfCorrectionRAG": CoTSelfCorrectionRAG,
    "ToTSelfCorrectionRAG": ToTSelfCorrectionRAG,
    "GoTSelfCorrectionRAG": GoTSelfCorrectionRAG,
}

def run_comparison(strategies_to_run: list, sample_size: int = None):
    """Run comparison for specified strategies."""
    
    # Load dataset
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "class_files_df.pkl")
    if not os.path.exists(dataset_path):
        dataset_path = os.path.join(os.getcwd(), "class_files_df.pkl")
    
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset not found at {dataset_path}")
        return
    
    df = pd.read_pickle(dataset_path)
    
    if sample_size and sample_size > 0:
        logger.info(f"Running on subset of {sample_size} samples")
        df = df.head(sample_size)
    else:
        logger.info(f"Running on full dataset of {len(df)} samples")
    
    all_metrics = []
    evaluator = RAGEvaluator()
    
    # Re-use the existing index if possible to save setup time
    # Note: CodeAware, Corrective, etc. expect specific index names usually.
    # To reuse 'rag-docstring', we might need to force it.
    default_index = config.index_names.get('simple', 'rag-docstring')
    
    for name in strategies_to_run:
        cls = STRATEGY_MAP.get(name)
        if not cls:
            logger.warning(f"Unknown strategy: {name}")
            continue
            
        try:
            logger.info(f"--- Running {name} ---")
            
            # Use specific index names if they are significantly different, 
            # otherwise allow reuse to save Pinecone limits.
            # Here we try to reuse 'rag-docstring' but heavily name-spaced.
            rag_system = cls(index_name=default_index, namespace=f"test-{name.lower()}")
            
            output_dir = os.path.join(config.paths.results_dir, f"comparison_{name}")
            os.makedirs(output_dir, exist_ok=True)
            
            # Create temp dataset file if using a subset
            current_dataset_path = dataset_path
            temp_file_path = None
            
            if sample_size and sample_size > 0:
                temp_file_path = os.path.join(output_dir, "temp_subset.pkl")
                df.to_pickle(temp_file_path)
                current_dataset_path = temp_file_path
                logger.info(f"Created temporary subset dataset at {temp_file_path}")

            start_time = time.time()
            try:
                # Capture results in case it returns early/crashes
                results_file = rag_system.process_dataset(dataset_path=current_dataset_path, output_dir=output_dir)
                total_time = time.time() - start_time
            except Exception as process_err:
                logger.error(f"Error processing dataset for {name}: {process_err}")
                if temp_file_path and os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                continue
            
            # Clean up temp file
            if temp_file_path and os.path.exists(temp_file_path):
                os.remove(temp_file_path)
            
            # Load results
            if not os.path.exists(results_file):
                 logger.error(f"Results file not found for {name}")
                 continue

            results_df = pd.read_pickle(results_file)
            if sample_size:
                results_df = results_df.head(sample_size)
                
            logger.info(f"Evaluating results for {name}...")
            evaluated_df = evaluator.evaluate_dataset(results_df)
            
            # Save evaluation
            evaluated_df.to_pickle(os.path.join(output_dir, f"{name}_evaluated.pkl"))
            evaluated_df.to_excel(os.path.join(output_dir, f"{name}_evaluated.xlsx"), index=False)
            
            # --- FULL CONSOLIDATION (Results + Eval + Costs) ---
            try:
                # Load costs
                cost_file = os.path.join(output_dir, f"{name}_costs.pkl")
                if os.path.exists(cost_file):
                    costs_df = pd.read_pickle(cost_file)
                    if sample_size: costs_df = costs_df.head(sample_size)
                    
                    # Prefix cost columns
                    costs_df.columns = [f"Cost_{c}" for c in costs_df.columns if not c.startswith("Cost_")]
                    
                    # Merge evaluated_df (which has results+metrics) with costs_df
                    full_consolidated_df = pd.concat([evaluated_df.reset_index(drop=True), costs_df.reset_index(drop=True)], axis=1)
                    
                    consolidated_path = os.path.join(output_dir, f"{name}_consolidated.xlsx")
                    full_consolidated_df.to_excel(consolidated_path, index=False)
                    logger.info(f"💾 Full consolidated report saved to {consolidated_path}")
            except Exception as e:
                logger.error(f"Failed to create full consolidated report: {e}")
            # ---------------------------------------------------
            
            # Metrics
            metrics = {
                "Method": name,
                "Total Time (s)": round(total_time, 2),
                "Avg Time/Sample (s)": round(total_time / len(results_df), 2),
                "Samples": len(results_df)
            }
            
            numeric_cols = evaluated_df.select_dtypes(include=['number']).columns
            for col in numeric_cols:
                if any(x in col.lower() for x in ['rouge', 'bleu', 'bert', 'faithfulness', 'score']):
                    metrics[col] = round(evaluated_df[col].mean(), 4)
            
            cost_file = os.path.join(output_dir, f"{name}_costs.pkl")
            if os.path.exists(cost_file):
                costs_df = pd.read_pickle(cost_file)
                if sample_size: costs_df = costs_df.head(sample_size)
                metrics["Avg API Calls"] = round(costs_df['api_calls'].mean(), 1)
            
            all_metrics.append(metrics)
            logger.info(f"Completed {name}")
            
        except Exception as e:
            logger.error(f"Failed to run {name}: {e}", exc_info=True)
            
    # Save Report
    if all_metrics:
        comparison_df = pd.DataFrame(all_metrics)
        report_path = os.path.join(config.paths.results_dir, "comprehensive_rag_comparison_report.csv")
        # Append to existing if exists? No, overwrite for clean run.
        comparison_df.to_csv(report_path, index=False)
        
        print("\n=== Comprehensive Comparison Report ===")
        print(comparison_df.to_string(index=False))
        print(f"\nReport saved to {report_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=["all", "simple", "self", "plain"], default="all")
    parser.add_argument("--structure", choices=["base", "cot", "tot", "got", "all"], default="all")
    parser.add_argument("--samples", type=int, default=None, help="Number of samples (default: all)")
    args = parser.parse_args()
    
    # Filter strategies
    selected_strategies = []
    
    groups = {
        "simple": ["SimpleRAG", "CoTRAG", "ToTRAG", "GoTRAG"],
        "self": ["SelfCorrectionRAG", "CoTSelfCorrectionRAG", "ToTSelfCorrectionRAG", "GoTSelfCorrectionRAG"],
        "plain": ["PlainLLM", "CoTPlainLLM", "ToTPlainLLM", "GoTPlainLLM"]
    }
    
    target_groups = groups.keys() if args.group == "all" else [args.group]
    
    for g in target_groups:
        for s in groups[g]:
            if args.structure == "all":
                selected_strategies.append(s)
            elif args.structure == "base" and "CoT" not in s and "ToT" not in s and "GoT" not in s:
                selected_strategies.append(s)
            elif args.structure == "cot" and "CoT" in s:
                selected_strategies.append(s)
            elif args.structure == "tot" and "ToT" in s:
                selected_strategies.append(s)
            elif args.structure == "got" and "GoT" in s:
                selected_strategies.append(s)
                
    print(f"Running strategies: {selected_strategies}")
    if not selected_strategies:
        print("No strategies matched selection.")
    else:
        run_comparison(selected_strategies, sample_size=args.samples)
