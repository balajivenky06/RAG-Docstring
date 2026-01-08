
import pandas as pd
import os

file_path = 'results/Consolidated.xlsx'

if not os.path.exists(file_path):
    print(f"Error: {file_path} does not exist.")
    exit(1)

try:
    df = pd.read_excel(file_path)
    print("Columns found:", df.columns.tolist())
    
    # Identify the 'Method' or 'Strategy' column
    method_col = None
    for col in ['Method', 'Strategy', 'model', 'approach']:
        if col in df.columns:
            method_col = col
            break
    
    if method_col:
        print(f"Grouping by: {method_col}")
        
        # Select numeric columns for aggregation
        numeric_cols = df.select_dtypes(include=['number']).columns
        
        # Filter for relevant metrics
        metrics = [c for c in numeric_cols if any(x in c.lower() for x in ['rouge', 'bleu', 'bert', 'score', 'time', 'cost', 'api'])]
        
        summary = df.groupby(method_col)[metrics].mean().sort_values(by='BERT_f1', ascending=False, key=lambda x: x if 'BERT_f1' in x.name else x) # Attempt to sort by BERT_f1 if exists
        
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        
        print("\n--- Summary Metrics by Method (Mean) ---")
        print(summary)
        
        print("\n--- Top Performer per Metric ---")
        for metric in metrics:
            try:
                # Higher is better for scores, lower is better for cost/time
                if any(x in metric.lower() for x in ['time', 'cost', 'calls']):
                    best = df.groupby(method_col)[metric].mean().idxmin()
                    val = df.groupby(method_col)[metric].mean().min()
                else:
                    best = df.groupby(method_col)[metric].mean().idxmax()
                    val = df.groupby(method_col)[metric].mean().max()
                print(f"{metric}: {best} ({val:.4f})")
            except:
                pass

    else:
        print("Could not find a 'Method' column to group by. Showing first 5 rows:")
        print(df.head())

except Exception as e:
    print(f"An error occurred: {e}")
