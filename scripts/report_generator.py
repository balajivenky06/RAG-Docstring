import os
import pickle
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(__file__)
RESULTS_DIR = os.path.join(os.path.dirname(BASE_DIR), "results")
COMP_DIR = os.path.join(RESULTS_DIR, "comparison")
REPORT_DIR = os.path.join(RESULTS_DIR, "report")

METHODS = ["SimpleRAG", "CodeAwareRAG", "CorrectiveRAG", "SelfCorrectionRAG", "FusionRAG"]

METRICS = [
    "rouge_1_f1",
    "bleu_score",
    "bert_score",
    "flesch_reading_ease",
    "conciseness",
    "parameter_coverage",
    "return_coverage",
    "exception_coverage",
    "faithfulness_score",
    "pydocstyle_adherence",
]


def _ensure_dirs():
    os.makedirs(REPORT_DIR, exist_ok=True)
    os.makedirs(os.path.join(REPORT_DIR, "figures"), exist_ok=True)
    # Increase default font sizes for publication-quality figures
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 14,
        "axes.labelsize": 12,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "legend.fontsize": 10
    })


def load_comparison_frames():
    macro = pd.read_csv(os.path.join(COMP_DIR, "macro_averages_with_deltas.csv"))
    paired = pd.read_csv(os.path.join(COMP_DIR, "paired_tests.csv"))
    per_sample = pd.read_csv(os.path.join(COMP_DIR, "per_sample_metrics_long.csv"))
    return macro, paired, per_sample


def load_costs() -> Dict[str, pd.DataFrame]:
    costs: Dict[str, pd.DataFrame] = {}
    for m in METHODS:
        pkl = os.path.join(RESULTS_DIR, f"{m}_costs.pkl")
        if os.path.exists(pkl):
            with open(pkl, "rb") as f:
                df = pickle.load(f)
            # normalize expected columns
            # Expect columns like: sample_id/index, execution_time_s, memory_mb, api_calls
            cols_lower = {c.lower(): c for c in df.columns}
            rename_map: Dict[str, str] = {}
            if "execution" in ",".join(cols_lower.keys()):
                for k, v in list(cols_lower.items()):
                    if "execution" in k and "time" in k:
                        rename_map[v] = "execution_time_s"
                        break
            if "memory" in cols_lower:
                rename_map[cols_lower["memory"]] = "memory_mb"
            if "api_calls" in cols_lower:
                rename_map[cols_lower["api_calls"]] = "api_calls"
            df = df.rename(columns=rename_map)
            costs[m] = df
    return costs


def bar_chart_macro(macro: pd.DataFrame):
    fig_dir = os.path.join(REPORT_DIR, "figures")
    for metric in METRICS:
        plt.figure(figsize=(8, 4))
        order = macro.sort_values(metric, ascending=False)
        plt.bar(order["rag_method"], order[metric], color="#4e79a7")
        plt.title(f"Macro average: {metric}")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, f"bar_{metric}.png"), dpi=600)
        plt.close()


def radar_chart_macro(macro: pd.DataFrame):
    # Normalize metrics 0-1 for radar
    df = macro.set_index("rag_method")[METRICS].copy()
    norm = (df - df.min()) / (df.max() - df.min()).replace({0: np.nan})
    norm = norm.fillna(0.0)
    labels = METRICS
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
    angles = np.concatenate([angles, [angles[0]]])
    fig_dir = os.path.join(REPORT_DIR, "figures")

    plt.figure(figsize=(7, 7))
    ax = plt.subplot(111, polar=True)
    for method, row in norm.iterrows():
        values = row.values
        values = np.concatenate([values, [values[0]]])
        ax.plot(angles, values, linewidth=1, label=method)
        ax.fill(angles, values, alpha=0.1)
    ax.set_thetagrids(angles[:-1] * 180 / np.pi, labels, fontsize=8)
    ax.set_title("Normalized radar across macro metrics")
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.1), fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "radar_macro.png"), dpi=600)
    plt.close()


def cost_vs_quality_scatter(macro: pd.DataFrame, costs: Dict[str, pd.DataFrame]):
    # Compute average costs per method
    rows = []
    for m, df in costs.items():
        avg_time = df.filter(regex="execution|time", axis=1).mean(axis=0).mean()
        avg_mem = df.filter(regex="memory", axis=1).mean(axis=0).mean()
        avg_calls = df.filter(regex="api_calls", axis=1).mean(axis=0).mean()
        rows.append({"rag_method": m, "avg_time_s": avg_time, "avg_mem_mb": avg_mem, "avg_api_calls": avg_calls})
    cost_df = pd.DataFrame(rows)
    merged = pd.merge(macro, cost_df, on="rag_method", how="left")

    fig_dir = os.path.join(REPORT_DIR, "figures")
    # Time vs ROUGE
    plt.figure(figsize=(6, 4))
    for _, r in merged.iterrows():
        plt.scatter(r["avg_time_s"], r["rouge_1_f1"], s=60, label=r["rag_method"])
        plt.text(r["avg_time_s"], r["rouge_1_f1"], r["rag_method"], fontsize=8, ha="left", va="bottom")
    plt.xlabel("Avg execution time (s)")
    plt.ylabel("ROUGE-1 F1 (macro)")
    plt.title("Cost vs Quality (Time vs ROUGE-1)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "scatter_time_vs_rouge.png"), dpi=600)
    plt.close()

    # API calls vs BERTScore
    plt.figure(figsize=(6, 4))
    for _, r in merged.iterrows():
        plt.scatter(r["avg_api_calls"], r["bert_score"], s=60, label=r["rag_method"])
        plt.text(r["avg_api_calls"], r["bert_score"], r["rag_method"], fontsize=8, ha="left", va="bottom")
    plt.xlabel("Avg API calls")
    plt.ylabel("BERTScore (macro)")
    plt.title("Cost vs Quality (API calls vs BERTScore)")
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "scatter_calls_vs_bert.png"), dpi=600)
    plt.close()


def annotate_significance_on_bars(macro: pd.DataFrame, paired: pd.DataFrame, metric: str, baseline: str = "SimpleRAG"):
    fig_dir = os.path.join(REPORT_DIR, "figures")
    order = macro.sort_values(metric, ascending=False)
    plt.figure(figsize=(8, 4))
    bars = plt.bar(order["rag_method"], order[metric], color="#59a14f")
    plt.title(f"Macro {metric} with significance vs {baseline}")
    plt.xticks(rotation=20)
    # pull p-values
    for i, method in enumerate(order["rag_method"]):
        if method == baseline:
            continue
        subset = paired[(paired["metric"] == metric)]
        # method_a-baseline or baseline-method_b
        pvals: List[float] = []
        pvals.extend(subset[(subset["method_a"] == method) & (subset["method_b"] == baseline)]["wilcoxon_p"].tolist())
        pvals.extend(subset[(subset["method_a"] == baseline) & (subset["method_b"] == method)]["wilcoxon_p"].tolist())
        p = pvals[0] if pvals else np.nan
        if pd.notna(p):
            y = bars[i].get_height()
            txt = "*" if p < 0.05 else ("†" if p < 0.1 else "ns")
            plt.text(i, y, f" {txt}", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, f"bar_{metric}_signif.png"), dpi=600)
    plt.close()


def write_markdown_report(macro: pd.DataFrame):
    path = os.path.join(REPORT_DIR, "REPORT.md")
    lines: List[str] = []
    lines.append("# Consolidated Comparison Report")
    lines.append("")
    lines.append("## Macro Averages (sorted by ROUGE-1 F1)")
    top = macro.sort_values("rouge_1_f1", ascending=False).reset_index(drop=True)
    # write a compact CSV-like section to avoid tabulate dependency
    header = ["rag_method", *METRICS]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for _, r in top.iterrows():
        row = [str(r["rag_method"])] + [f"{r[m]:.4g}" if pd.notna(r[m]) else "" for m in METRICS]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")
    lines.append("## Figures")
    for m in METRICS:
        lines.append(f"- Macro bar: figures/bar_{m}.png")
    lines.append("- Radar: figures/radar_macro.png")
    lines.append("- Time vs ROUGE: figures/scatter_time_vs_rouge.png")
    lines.append("- API calls vs BERTScore: figures/scatter_calls_vs_bert.png")
    lines.append("")
    lines.append("## Notes")
    lines.append("- Bars with '*' differ from SimpleRAG at p<0.05 by Wilcoxon; '†' at p<0.1; 'ns' not significant.")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    _ensure_dirs()
    macro, paired, _ = load_comparison_frames()
    bar_chart_macro(macro)
    radar_chart_macro(macro)
    costs = load_costs()
    cost_vs_quality_scatter(macro, costs)
    # significance annotations for a few key metrics
    for metric in ["rouge_1_f1", "bert_score", "faithfulness_score"]:
        annotate_significance_on_bars(macro, paired, metric, baseline="SimpleRAG")
    write_markdown_report(macro)
    print(f"Report written to {REPORT_DIR}")


if __name__ == "__main__":
    main()


