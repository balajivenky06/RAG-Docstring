
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

# Set style for publication quality
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")
# OUTPUT_DIR will be set at runtime based on the target folder
OUTPUT_DIR = ""

# Define global color palettes for consistency
PALETTE = {"RAG": "#2ecc71", "Plain LLM": "#95a5a6", "Self-Correction": "#e67e22"}
MARKERS = {"Base": "o", "CoT": "^", "ToT": "X", "GoT": "s"}

def load_data(results_dir="results"):
    files = {
        "Plain LLM": os.path.join(results_dir, "comprehensive_plain_comparison_report.csv"),
        "RAG": os.path.join(results_dir, "comprehensive_rag_comparison_report.csv"),
        "Self-Correction": os.path.join(results_dir, "comprehensive_selfcorrectiverag_comparison_report.csv")
    }
    
    dfs = []
    for family, path in files.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['Family'] = family
            dfs.append(df)
            
    if not dfs:
        print("No strategy comparison reports found. Please run compare_all_strategies.py first.")
        return pd.DataFrame()
        
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Normalize Method names for cleaner legend
    # Remove 'PlainLLM', 'RAG', 'SelfCorrectionRAG' suffixes if present to just show 'Base', 'CoT', 'ToT', 'GoT'
    def clean_method(name):
        if "CoT" in name: return "CoT"
        if "ToT" in name: return "ToT"
        if "GoT" in name: return "GoT"
        return "Base"
        
    full_df['Reasoning Mode'] = full_df['Method'].apply(clean_method)
    return full_df

def plot_faithfulness_vs_latency(df):
    plt.figure(figsize=(14, 9))
    
    # Check if 'faithfulness_score' and 'Avg Time/Sample (s)' exist
    if 'faithfulness_score' not in df.columns or 'Avg Time/Sample (s)' not in df.columns:
        print("Required columns for Pareto Frontier not found. Skipping.")
        return
    
    # Define custom palette for Families to make them distinct
    # RAG = High contrast Green (Winner)
    # Plain = Gray (Baseline)
    # SelfCorrect = Orange (Alternative)
    palette = {"RAG": "#2ecc71", "Plain LLM": "#95a5a6", "Self-Correction": "#e67e22"}
    markers = {"Base": "o", "CoT": "^", "ToT": "X", "GoT": "s"} # Distinct shapes
    
    # Main Scatter
    ax = sns.scatterplot(
        data=df,
        x="Avg Time/Sample (s)",
        y="faithfulness_score",
        hue="Family",
        style="Reasoning Mode",
        markers=markers,
        palette=palette,
        s=400, # Big chunky points
        alpha=0.9,
        edgecolor="black",
        linewidth=1.5
    )
    
    # Add Grid
    ax.grid(True, which="both", ls="-", alpha=0.15)
    
    # Log Scale X
    ax.set_xscale('log')
    
    # Add Labels with manual collision avoidance
    # Map: Method -> (x_multiplier, y_additive)
    offset_map = {
        "SimpleRAG": (1.1, 0.01),      # Top Right (Winner)
        "CoTRAG": (1.1, -0.01),        # Bottom Right
        "ToTRAG": (1.1, 0.0),          # Right
        "GoTRAG": (1.1, 0.0),          # Right
        
        "PlainLLM": (0.85, -0.015),    # Bottom Left
        "CoTPlainLLM": (0.85, 0.015),  # Top Left
        "ToTPlainLLM": (1.1, 0.0),     # Right
        "GoTPlainLLM": (1.1, 0.0),     # Right

        "SelfCorrectionRAG": (1.1, 0.01),     # Top Right
        "CoTSelfCorrectionRAG": (1.1, -0.01), # Bottom Right
        "ToTSelfCorrectionRAG": (1.1, 0.0),   # Right
        "GoTSelfCorrectionRAG": (1.1, 0.0),   # Right
    }

    for i, row in df.iterrows():
        label = row['Method']
        x = row['Avg Time/Sample (s)']
        y = row['faithfulness_score']
        
        # Get offsets or default
        x_mult, y_add = offset_map.get(label, (1.1, 0.0))
        
        if "SimpleRAG" in label:
            label = "★ " + label # Highlight winner
            
        plt.text(x * x_mult, y + y_add, label, fontsize=10, weight='bold', alpha=0.9)

    plt.title("Efficiency Frontier: Faithfulness vs. Latency", fontsize=20, weight='bold', pad=20)
    plt.xlabel("Latency per Sample (Seconds) - Log Scale", fontsize=15)
    plt.ylabel("Faithfulness Score (LLM-Judge)", fontsize=15)
    
    # Legend improvements
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title="Strategy Group", frameon=True, shadow=True, fontsize=12, title_fontsize=13)
    
    # Annotate the "Efficiency Gap" (Arrow from SimpleRAG to ToTRAG)
    try:
        rag_row = df[df['Method'] == 'SimpleRAG'].iloc[0]
        tot_row = df[df['Method'] == 'ToTRAG'].iloc[0]
        
        plt.annotate(
            "15x Slower &\nLess Faithful", 
            xy=(tot_row['Avg Time/Sample (s)'], tot_row['faithfulness_score']), 
            xytext=(rag_row['Avg Time/Sample (s)'] * 2, rag_row['faithfulness_score'] - 0.08),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=-0.2", color="#c0392b", lw=2.5, ls='--'),
            color="#c0392b", fontsize=12, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#c0392b", alpha=0.9)
        )
    except IndexError:
        pass # Skip annotation if methods not found

    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/tradeoff_faithfulness_latency_refined.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_faithfulness_vs_api_calls(df):
    plt.figure(figsize=(14, 9))
    
    if 'faithfulness_score' not in df.columns or 'Avg API Calls' not in df.columns:
        print("Required columns for API calls Chart not found. Skipping.")
        return
    
    palette = {"RAG": "#2ecc71", "Plain LLM": "#95a5a6", "Self-Correction": "#e67e22"}
    markers = {"Base": "o", "CoT": "^", "ToT": "X", "GoT": "s"}
    
    ax = sns.scatterplot(
        data=df,
        x="Avg API Calls",
        y="faithfulness_score",
        hue="Family",
        style="Reasoning Mode",
        markers=markers,
        palette=palette,
        s=400,
        alpha=0.9,
        edgecolor="black",
        linewidth=1.5
    )
    
    ax.grid(True, which="both", ls="-", alpha=0.15)
    
    offset_map = {
        "SimpleRAG": (0.2, 0.01),      
        "CoTRAG": (0.2, -0.01),        
        "ToTRAG": (0.2, 0.01),          
        "GoTRAG": (0.2, 0.01),          
        "PlainLLM": (-0.2, -0.015),    
        "CoTPlainLLM": (-0.2, 0.015),  
        "ToTPlainLLM": (0.2, 0.01),     
        "GoTPlainLLM": (0.2, 0.01),     
        "SelfCorrectionRAG": (0.2, 0.01),     
        "CoTSelfCorrectionRAG": (0.2, -0.01), 
        "ToTSelfCorrectionRAG": (0.2, 0.01),   
        "GoTSelfCorrectionRAG": (0.2, 0.01),   
    }

    for i, row in df.iterrows():
        label = row['Method']
        x = row['Avg API Calls']
        y = row['faithfulness_score']
        
        x_add, y_add = offset_map.get(label, (0.2, 0.01))
        
        if "SimpleRAG" in label:
            label = "★ " + label
            
        plt.text(x + 0.1 + x_add, y + y_add, label, fontsize=10, weight='bold', alpha=0.9)

    plt.title("Computational Cost: Faithfulness vs. API Calls", fontsize=20, weight='bold', pad=20)
    plt.xlabel("Average Number of API Calls per Sample", fontsize=15)
    plt.ylabel("Faithfulness Score (LLM-Judge)", fontsize=15)
    
    plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0, title="Strategy Group", frameon=True, shadow=True, fontsize=12, title_fontsize=13)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/tradeoff_faithfulness_api_calls.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_faithfulness_bar(df):
    plt.figure(figsize=(12, 6))
    
    # Order: Plain -> RAG -> SelfCorrection
    order = ["Plain LLM", "RAG", "Self-Correction"]
    
    g = sns.barplot(
        data=df,
        x="Family",
        y="faithfulness_score",
        hue="Reasoning Mode",
        palette="viridis",
        order=order,
        hue_order=["Base", "CoT", "ToT", "GoT"],
        errorbar=None
    )
    
    # Add value labels
    for container in g.containers:
        g.bar_label(container, fmt='%.2f', padding=3)

    plt.title("Faithfulness Score by Strategy & Reasoning Mode", fontsize=16, weight='bold', pad=20)
    plt.ylabel("Faithfulness Score", fontsize=12)
    plt.xlabel("Architecture Family", fontsize=12)
    plt.ylim(0.4, 0.8) # Zoom in on the relevant range
    plt.legend(title="Reasoning Variant", bbox_to_anchor=(1.01, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/comparison_faithfulness_bar.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_nlp_metrics_comparison(df):
    metrics = ['rouge_1_f1', 'bert_score', 'faithfulness_score']
    available_metrics = [m for m in metrics if m in df.columns]
    
    if not available_metrics:
        print("Required columns for NLP metrics comparison not found.")
        return
        
    df_melted = df.melt(id_vars=['Method', 'Family'], value_vars=available_metrics, var_name='Metric', value_name='Score')
    
    metric_map = {
        'rouge_1_f1': 'ROUGE-1 (F1)',
        'bert_score': 'BERTScore',
        'faithfulness_score': 'Faithfulness'
    }
    df_melted['Metric'] = df_melted['Metric'].map(metric_map)
    
    # Select baseline architectures to keep chart clean
    base_methods = ['PlainLLM', 'SimpleRAG', 'SelfCorrectionRAG']
    df_base = df_melted[df_melted['Method'].isin(base_methods)].copy()
    
    if df_base.empty:
        return
        
    df_base['Method'] = pd.Categorical(df_base['Method'], categories=base_methods, ordered=True)
    
    plt.figure(figsize=(12, 6))
    
    g = sns.barplot(
        data=df_base,
        x="Method",
        y="Score",
        hue="Metric",
        palette="mako"
    )
    
    for container in g.containers:
        g.bar_label(container, fmt='%.3f', padding=3)

    plt.title("NLP Evaluation Metrics Across Baseline Architectures", fontsize=16, weight='bold', pad=20)
    plt.ylabel("Score (0.0 - 1.0)", fontsize=12)
    plt.xlabel("Architecture", fontsize=12)
    plt.ylim(0, 1.0)
    plt.legend(title="Metric", bbox_to_anchor=(1.01, 1), loc='upper left')
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/nlp_metrics_comparison.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_human_vs_judge_correlation(results_dir="results"):
    human_eval_path = os.path.join(results_dir, "human_eval", "human_eval_sheet.xlsx")
    if not os.path.exists(human_eval_path):
        print(f"Human eval sheet not found at {human_eval_path}. Skipping correlation chart.")
        return
        
    df = pd.read_excel(human_eval_path)
    
    # Support mapping the old categorical 'Faithfulness_Label' if the new column doesn't exist yet
    target_col = 'HUMAN_Faithfulness_Score (0.0 to 1.0)'
    if target_col not in df.columns:
        if 'Faithfulness_Label' in df.columns:
            print("Detected old categorical Faithfulness_Label in Human Eval Sheet. Converting 'Supported'/'Unsupported' to 1.0/0.0 for correlation...")
            df[target_col] = df['Faithfulness_Label'].map({"Supported": 1.0, "Partially Supported": 0.5, "Unsupported": 0.0})
        else:
            print("Human scores are not filled out yet. Skipping correlation chart.")
            return

    if df[target_col].isnull().all():
        print("Human scores are not filled out yet. Skipping correlation chart.")
        return
        
    # Drop rows where human hasn't scored yet
    df = df.dropna(subset=[target_col])
    
    plt.figure(figsize=(8, 8))
    sns.regplot(
        data=df, 
        x='LLM_Faithfulness_Score', 
        y=target_col,
        scatter_kws={'alpha': 0.6, 's': 100, 'color': '#3498db'},
        line_kws={'color': '#e74c3c', 'linewidth': 3, 'label': 'Trend Line'}
    )
    
    # Calculate correlation coefficient
    corr = df['LLM_Faithfulness_Score'].corr(df[target_col])
    plt.annotate(f"Pearson r = {corr:.2f}", xy=(0.05, 0.95), xycoords='axes fraction', 
                 fontsize=14, weight='bold', bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9))
    
    plt.title("LLM Judge vs Human Expert Correlation", fontsize=18, weight='bold', pad=20)
    plt.xlabel("LLM Judge Faithfulness Score", fontsize=14)
    plt.ylabel("Human Expert Faithfulness Score", fontsize=14)
    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/human_judge_correlation.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_performance_by_complexity(results_dir="results"):
    # We look for the evaluated file of SimpleRAG that was tagged with complexity
    file_path = os.path.join(results_dir, "comparison_SimpleRAG", "SimpleRAG_evaluated.pkl")
    if not os.path.exists(file_path):
        print("Complexity data not found. Skipping complexity chart.")
        return
        
    df = pd.read_pickle(file_path)
    if 'OOP_Complexity' not in df.columns:
        print("OOP_Complexity column not found in SimpleRAG_evaluated.pkl. Did you run analyze_complexity_traces.py?")
        return
        
    plt.figure(figsize=(10, 6))
    
    order = ["Simple", "Moderate", "Complex"]
    sns.boxplot(
        data=df,
        x="OOP_Complexity",
        y="faithfulness_score",
        order=order,
        palette="Blues"
    )
    
    plt.title("SimpleRAG Faithfulness by Code Complexity", fontsize=18, weight='bold', pad=20)
    plt.xlabel("OOP Class Complexity", fontsize=14)
    plt.ylabel("Faithfulness Score", fontsize=14)
    
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/complexity_stratified_performance.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_efficiency_heatmap(df):
    # Prepare pivot table for heatmap
    # Efficiency = Faithfulness / log(Time) ? Or just pure Faithfulness
    
    # Let's simple plot Cost (Time) vs Accuracy (Faithfulness) in a dual-axis chart for the WINNING family (RAG)
    rag_df = df[df['Family'] == 'RAG'].copy()
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Reasoning Mode', fontsize=12)
    ax1.set_ylabel('Faithfulness (Higher is Better)', color=color, fontsize=12)
    sns.lineplot(data=rag_df, x='Reasoning Mode', y='faithfulness_score', ax=ax1, color=color, marker='o', markersize=10, linewidth=3)
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(False)
    
    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    
    color = 'tab:red'
    ax2.set_ylabel('Latency (s) (Lower is Better)', color=color, fontsize=12)  # we already handled the x-label with ax1
    sns.barplot(data=rag_df, x='Reasoning Mode', y='Avg Time/Sample (s)', ax=ax2, color=color, alpha=0.3)
    ax2.tick_params(axis='y', labelcolor=color)
    ax2.grid(False)
    
    # Add values to bars
    for i, v in enumerate(rag_df['Avg Time/Sample (s)']):
        ax2.text(i, v + 1, f"{v:.1f}s", color='red', ha='center', fontweight='bold')

    plt.title("RAG Family: The Cost of Reasoning", fontsize=16, weight='bold', pad=20)
    fig.tight_layout()
    plt.savefig(f"{OUTPUT_DIR}/rag_efficiency_dual_axis.png", dpi=300)
    plt.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="results", help="Directory containing model results (e.g. results/llama3_latest)")
    args = parser.parse_args()
    
    OUTPUT_DIR = os.path.join(args.dir, "visualization/paper_charts")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Loading data from {args.dir}...")
    df = load_data(args.dir)
    print(f"Data Loaded: {len(df)} strategies.")
    if len(df) > 0:
        print(df[['Method', 'Family', 'Reasoning Mode', 'faithfulness_score', 'Avg Time/Sample (s)']])
    
    print("Generating Chart 1: Trade-off Scatter (Latency)...")
    plot_faithfulness_vs_latency(df)
    
    print("Generating Chart 2: Trade-off Scatter (API Calls)...")
    plot_faithfulness_vs_api_calls(df)
    
    print("Generating Chart 3: Comparative Bar (Faithfulness)...")
    plot_faithfulness_bar(df)
    
    print("Generating Chart 4: NLP Metrics Comparison...")
    plot_nlp_metrics_comparison(df)
    
    print("Generating Chart 5: Efficiency Dual Axis...")
    plot_efficiency_heatmap(df)
    
    print("Generating Chart 6: Human vs Judge Correlation...")
    plot_human_vs_judge_correlation(args.dir)
    
    print("Generating Chart 7: Complexity Breakdown...")
    plot_performance_by_complexity(args.dir)
    
    print(f"Done! Check {OUTPUT_DIR}")
