import pandas as pd
import os

file_path = 'results/Consolidated.xlsx'
# Read with header at row 4 (0-indexed)
df = pd.read_excel(file_path, header=4)

# Rename identifier columns
df.rename(columns={'Unnamed: 0': 'Category', 'Unnamed: 1': 'Strategy'}, inplace=True)

# Forward fill the Category column (handling merged cells)
df['Category'] = df['Category'].fillna(method='ffill')

# Drop rows where Strategy is NaN (likely separator rows)
df = df.dropna(subset=['Strategy'])

# List of metric columns
metric_cols = [
    'rouge_1_f1', 'bleu_score', 'bert_score', 'flesch_reading_ease',
    'conciseness', 'parameter_coverage', 'return_coverage', 
    'exception_coverage', 'faithfulness_score'
]

# List of cost columns
cost_cols = [
    'Cost_execution_time', 'Cost_api_calls', 
    'Cost_retrieval_time', 'Cost_generation_time'
]

all_cols = metric_cols + cost_cols

# Ensure numeric
for col in all_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Create a combined name for clearer display
df['Method_Full'] = df['Category'].astype(str) + " - " + df['Strategy'].astype(str)

# Select relevant columns for the report
report_cols = ['Method_Full'] + [c for c in all_cols if c in df.columns]
report_df = df[report_cols].copy()

# Sort by a key metric, e.g., bert_score (descending)
if 'bert_score' in report_df.columns:
    report_df = report_df.sort_values(by='bert_score', ascending=False)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.4f}'.format)

print("\n=== EXPERIMENT METRICS FINDINGS ===")
print(report_df.to_string(index=False))

print("\n\n=== KEY OBSERVATIONS ===")

# Best for Quality (BERT Score)
if 'bert_score' in df.columns:
    best_bert = df.loc[df['bert_score'].idxmax()]
    print(f"Top Quality (BERT Score): {best_bert['Method_Full']} ({best_bert['bert_score']:.4f})")

# Best for Faithfulness
if 'faithfulness_score' in df.columns:
    best_faith = df.loc[df['faithfulness_score'].idxmax()]
    print(f"Top Faithfulness: {best_faith['Method_Full']} ({best_faith['faithfulness_score']:.4f})")

# Lowest Latency
if 'Cost_execution_time' in df.columns:
    fastest = df.loc[df['Cost_execution_time'].idxmin()]
    print(f"Fastest Execution: {fastest['Method_Full']} ({fastest['Cost_execution_time']:.4f}s)")

# Best Coverage (Parameter)
if 'parameter_coverage' in df.columns:
    best_cov = df.loc[df['parameter_coverage'].idxmax()]
    print(f"Best Parameter Coverage: {best_cov['Method_Full']} ({best_cov['parameter_coverage']:.4f})")

