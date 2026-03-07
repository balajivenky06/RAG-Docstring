
import os
import sys
import pandas as pd

# Add the current directory to Python path
# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from rag_system.cost_analyzer import CostAnalyzer

def main():
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    
    # Define expected strategies
    strategies = [
        "SimpleRAG", "CodeAwareRAG", "CorrectiveRAG", "FusionRAG", "SelfCorrectionRAG",
        "CoTRAG", "ToTRAG", "GoTRAG",
        "CoTSelfCorrectionRAG", "ToTSelfCorrectionRAG", "GoTSelfCorrectionRAG"
    ]
    
    cost_files = {}
    for strategy in strategies:
        # Check root results dir
        direct_path = os.path.join(results_dir, f"{strategy}_costs.pkl")
        if os.path.exists(direct_path):
            cost_files[strategy] = direct_path
        else:
            # Check subdirectory
            subdir_path = os.path.join(results_dir, f"comparison_{strategy}", f"{strategy}_costs.pkl")
            if os.path.exists(subdir_path):
                cost_files[strategy] = subdir_path
    
    if not cost_files:
        print("No cost files found in results directory.")
        return

    analyzer = CostAnalyzer()
    analyzer.load_cost_data(cost_files)
    
    # Generate report
    report_path = os.path.join(results_dir, "cost_analysis_report.txt")
    analyzer.generate_cost_report(report_path)
    
    # Generate chart
    chart_path = os.path.join(results_dir, "cost_comparison.png")
    analyzer.create_cost_comparison_chart(chart_path)
    
    print("Cost analysis completed.")

if __name__ == "__main__":
    main()
