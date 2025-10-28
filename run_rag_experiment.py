"""
Main script to run RAG experiments with dynamic configuration.
"""

import pandas as pd
import os
import argparse
import sys
from typing import Dict, List, Optional
import logging

# Add the current directory to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from rag_system import (
    SimpleRAG, CodeAwareRAG, CorrectiveRAG, FusionRAG, SelfRAG,
    RAGEvaluator, CostAnalyzer, RAGVisualizer,
    config, get_index_name, get_index_namespace, get_knowledge_base_urls
)

def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, config.logging.level),
        format=config.logging.format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(config.logging.log_file)
        ]
    )
    return logging.getLogger(__name__)

def load_config_from_args(args):
    """Load configuration from command line arguments."""
    if args.pinecone_api_key:
        config.pinecone.api_key = args.pinecone_api_key
    
    if args.generator_model:
        config.model.generator_model = args.generator_model
    
    if args.helper_model:
        config.model.helper_model = args.helper_model
    
    if args.temperature:
        config.model.temperature = args.temperature
    
    if args.top_k:
        config.retrieval.top_k = args.top_k
    
    if args.output_dir:
        config.paths.results_dir = args.output_dir
        config.paths.evaluation_dir = os.path.join(args.output_dir, "evaluation")
        config.paths.visualization_dir = os.path.join(args.output_dir, "visualizations")

def get_rag_methods(method_names: List[str]) -> Dict[str, dict]:
    """Get RAG method configurations."""
    rag_methods = {
        "simple": {
            "class": SimpleRAG,
            "index_name": get_index_name('simple'),
            "namespace": get_index_namespace('simple')
        },
        "code_aware": {
            "class": CodeAwareRAG,
            "index_name": get_index_name('code_aware'),
            "namespace": get_index_namespace('code_aware')
        },
        "corrective": {
            "class": CorrectiveRAG,
            "index_name": get_index_name('corrective'),
            "namespace": get_index_namespace('corrective')
        },
        "fusion": {
            "class": FusionRAG,
            "index_name": get_index_name('fusion'),
            "namespace": get_index_namespace('fusion')
        },
        "self": {
            "class": SelfRAG,
            "index_name": get_index_name('self'),
            "namespace": get_index_namespace('self')
        }
    }
    
    # Filter methods based on requested methods
    if method_names:
        filtered_methods = {}
        for method_name in method_names:
            if method_name in rag_methods:
                filtered_methods[method_name] = rag_methods[method_name]
            else:
                print(f"Warning: Unknown RAG method '{method_name}'")
        return filtered_methods
    
    return rag_methods

def run_single_method(method_name: str, dataset_path: str, logger: logging.Logger) -> Optional[str]:
    """Run a single RAG method."""
    rag_methods = get_rag_methods([method_name])
    
    if not rag_methods:
        logger.error(f"No valid RAG method found for '{method_name}'")
        return None
    
    method_config = rag_methods[method_name]
    logger.info(f"Running {method_name} RAG method...")
    
    try:
        # Initialize RAG instance
        rag_instance = method_config["class"](
            index_name=method_config["index_name"],
            namespace=method_config["namespace"]
        )
        
        # Process dataset
        results_file = rag_instance.process_dataset(dataset_path)
        
        # Load results for evaluation
        results_df = pd.read_pickle(results_file)
        
        # Evaluate docstrings
        evaluator = RAGEvaluator()
        evaluated_df = evaluator.evaluate_dataset(results_df)
        
        # Save evaluated results
        eval_file = os.path.join(config.paths.evaluation_dir, f"{method_name}_evaluation.pkl")
        evaluated_df.to_pickle(eval_file)
        
        # Save evaluated results as Excel file
        eval_excel_file = os.path.join(config.paths.evaluation_dir, f"{method_name}_evaluation.xlsx")
        evaluated_df.to_excel(eval_excel_file, index=False, engine='openpyxl')
        
        logger.info(f"{method_name} RAG completed successfully. Results saved to {results_file}")
        logger.info(f"Evaluation saved to {eval_file}")
        logger.info(f"Evaluation saved to {eval_excel_file}")
        
        return results_file
        
    except Exception as e:
        logger.error(f"Error running {method_name} RAG: {e}")
        return None

def run_multiple_methods(method_names: List[str], dataset_path: str, logger: logging.Logger):
    """Run multiple RAG methods and compare results."""
    rag_methods = get_rag_methods(method_names)
    
    if not rag_methods:
        logger.error("No valid RAG methods found")
        return
    
    logger.info(f"Running {len(rag_methods)} RAG methods: {list(rag_methods.keys())}")
    
    all_results = []
    all_costs = []
    
    # Initialize evaluator and visualizer
    evaluator = RAGEvaluator()
    visualizer = RAGVisualizer(output_dir=config.paths.visualization_dir)
    
    for method_name, method_config in rag_methods.items():
        logger.info(f"\n--- Running {method_name} ---")
        
        try:
            # Initialize RAG instance
            rag_instance = method_config["class"](
                index_name=method_config["index_name"],
                namespace=method_config["namespace"]
            )
            
            # Process dataset
            results_file = rag_instance.process_dataset(dataset_path)
            
            # Load results for evaluation
            results_df = pd.read_pickle(results_file)
            
            # Evaluate docstrings
            evaluated_df = evaluator.evaluate_dataset(results_df)
            
            # Save evaluated results
            eval_file = os.path.join(config.paths.evaluation_dir, f"{method_name}_evaluation.pkl")
            evaluated_df.to_pickle(eval_file)
            
            # Save evaluated results as Excel file
            eval_excel_file = os.path.join(config.paths.evaluation_dir, f"{method_name}_evaluation.xlsx")
            evaluated_df.to_excel(eval_excel_file, index=False, engine='openpyxl')
            
            logger.info(f"{method_name} completed. Results saved to {results_file}")
            logger.info(f"Evaluation saved to {eval_file}")
            logger.info(f"Evaluation saved to {eval_excel_file}")
            
            all_results.append(evaluated_df)
            
        except Exception as e:
            logger.error(f"Error running {method_name}: {e}")
            continue
    
    if all_results:
        # Combine all results
        combined_results = pd.concat(all_results, ignore_index=True)
        combined_file = os.path.join(config.paths.results_dir, "all_rag_results_combined.pkl")
        combined_results.to_pickle(combined_file)
        
        # Save combined results as Excel file
        combined_excel_file = os.path.join(config.paths.results_dir, "all_rag_results_combined.xlsx")
        combined_results.to_excel(combined_excel_file, index=False, engine='openpyxl')
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        visualizer.create_performance_comparison_chart(combined_results)
        visualizer.create_radar_chart(combined_results)
        visualizer.create_box_plots(combined_results)
        visualizer.create_performance_heatmap(combined_results)
        
        # Generate summary report
        summary_file = os.path.join(config.paths.evaluation_dir, "summary_report.txt")
        evaluator.generate_summary_report(combined_results, summary_file)
        
        logger.info(f"All methods completed. Combined results saved to {combined_file}")
        logger.info(f"Combined results saved to {combined_excel_file}")
        logger.info(f"Summary report saved to {summary_file}")
        logger.info(f"Summary Excel report saved to {summary_file.replace('.txt', '.xlsx')}")
    else:
        logger.error("No methods completed successfully")

def main():
    """Main function to run RAG experiments."""
    parser = argparse.ArgumentParser(description="Run RAG-based docstring generation experiments")
    
    # Required arguments
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to the dataset pickle file")
    
    # Optional arguments
    parser.add_argument("--pinecone_api_key", type=str,
                       help="Pinecone API key (default: from config)")
    
    # Optional arguments
    parser.add_argument("--methods", nargs="+", 
                       choices=["simple", "code_aware", "corrective", "fusion", "self"],
                       help="RAG methods to run (default: all)")
    parser.add_argument("--single_method", type=str,
                       choices=["simple", "code_aware", "corrective", "fusion", "self"],
                       help="Run only a single RAG method")
    parser.add_argument("--generator_model", type=str,
                       help="Generator model name")
    parser.add_argument("--helper_model", type=str,
                       help="Helper model name")
    parser.add_argument("--temperature", type=float,
                       help="Temperature for generation")
    parser.add_argument("--top_k", type=int,
                       help="Number of top results to retrieve")
    parser.add_argument("--output_dir", type=str,
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Setup logging
    logger = setup_logging()
    
    # Load configuration from arguments
    load_config_from_args(args)
    
    # Validate dataset path
    if not os.path.exists(args.dataset_path):
        logger.error(f"Dataset file not found: {args.dataset_path}")
        return 1
    
    # Create output directories
    os.makedirs(config.paths.results_dir, exist_ok=True)
    os.makedirs(config.paths.evaluation_dir, exist_ok=True)
    os.makedirs(config.paths.visualization_dir, exist_ok=True)
    
    try:
        if args.single_method:
            # Run single method
            result_file = run_single_method(args.single_method, args.dataset_path, logger)
            if result_file:
                logger.info(f"Single method experiment completed successfully")
                return 0
            else:
                logger.error("Single method experiment failed")
                return 1
        else:
            # Run multiple methods
            method_names = args.methods if args.methods else ["simple", "code_aware", "corrective", "fusion", "self"]
            run_multiple_methods(method_names, args.dataset_path, logger)
            logger.info("Multi-method experiment completed successfully")
            return 0
            
    except Exception as e:
        logger.error(f"Experiment failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)