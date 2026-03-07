import os
import pandas as pd
import ast
import argparse

def analyze_oop_complexity(code: str) -> str:
    """Analyze Python code and classify its OOP complexity."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return 'Unknown'
        
    class_count = 0
    method_count = 0
    method_max = 0
    has_inheritance = False
    
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            class_count += 1
            if len(node.bases) > 0 and getattr(node.bases[0], 'id', '') != 'object':
                has_inheritance = True
                
            methods = [n for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
            num_methods = len(methods)
            method_count += num_methods
            if num_methods > method_max:
                method_max = num_methods
                
    if class_count == 0:
        return 'Not OOP'
        
    if has_inheritance or method_max > 8:
        return 'Complex'
    elif method_max >= 4:
        return 'Moderate'
    else:
        return 'Simple'

def analyze_traces_and_complexity(results_dir: str):
    print(f"Analyzing OOP complexity in {results_dir}...")
    
    # Process all evaluated pkl files
    for root, _, files in os.walk(results_dir):
        for file in files:
            if file.endswith("_evaluated.pkl"):
                path = os.path.join(root, file)
                df = pd.read_pickle(path)
                
                if 'code' not in df.columns and 'Code_without_comments' not in df.columns:
                    continue
                    
                code_col = 'code' if 'code' in df.columns else 'Code_without_comments'
                
                # Classify complexity
                df['OOP_Complexity'] = df[code_col].apply(analyze_oop_complexity)
                
                # If it's a reasoning model, calculate trace length
                if 'generated_docstring' in df.columns:
                    # Often traces are stripped in final output, but if raw output is retained we can match it.
                    # As a proxy for this benchmark, we look at the 'reasoning_length' if we saved it in costs, 
                    # or purely the output length if it includes scratchpads.
                    df['Output_Length'] = df['generated_docstring'].apply(len)
                
                # Resave
                df.to_pickle(path)
                
                # Aggregate and display
                if 'faithfulness_score' in df.columns:
                    summary = df.groupby('OOP_Complexity')['faithfulness_score'].mean().round(3)
                    print(f"\n--- {file} ---")
                    print(summary)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="results", help="Directory containing results to scan.")
    args = parser.parse_args()
    
    analyze_traces_and_complexity(args.dir)
