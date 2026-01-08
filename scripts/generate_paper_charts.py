
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import numpy as np

# Set style for publication quality
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.5)
sns.set_style("whitegrid")

OUTPUT_DIR = "results/visualization/paper_charts"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def load_data():
    files = {
        "Plain LLM": "results/comprehensive_plain_comparison_report.csv",
        "RAG": "results/comprehensive_rag_comparison_report.csv",
        "Self-Correction": "results/comprehensive_selfcorrectiverag_comparison_report.csv"
    }
    
    dfs = []
    for family, path in files.items():
        if os.path.exists(path):
            df = pd.read_csv(path)
            df['Family'] = family
            dfs.append(df)
            
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
    plt.figure(figsize=(14, 9)) # Larger size
    
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
    print("Loading data...")
    df = load_data()
    print(f"Data Loaded: {len(df)} strategies.")
    print(df[['Method', 'Family', 'Reasoning Mode', 'faithfulness_score', 'Avg Time/Sample (s)']])
    
    print("Generating Chart 1: Trade-off Scatter...")
    plot_faithfulness_vs_latency(df)
    
    print("Generating Chart 2: Comparative Bar...")
    plot_faithfulness_bar(df)
    
    print("Generating Chart 3: Efficiency Dual Axis...")
    plot_efficiency_heatmap(df)
    
    print(f"Done! Check {OUTPUT_DIR}")
